import json
import os
import shutil
import tempfile
import time
import uuid
from importlib import util as importlib_util
from pathlib import Path, PurePath

import fsspec
import pooch
import pytest

OZ = pooch.os_cache("test_data")  # Path("/Users/utz.ermel/Documents/copick/testproject")  # pooch.os_cache("test_data")
TOTO = pooch.create(
    path=OZ,
    base_url="doi:10.5281/zenodo.11238625",
    registry={
        "sample_project.zip": "md5:b06ee6f4541b1ad06c988c8ca8d09945",
    },
)

# Determine if all tests should be run
RUN_ALL = bool(int(os.environ.get("RUN_ALL", 1)))

CLEANUP = True


@pytest.fixture(scope="session")
def local_path() -> Path:
    TOTO.fetch("sample_project.zip", processor=pooch.Unzip(extract_dir="sample_project"))
    yield OZ / "sample_project"

    if CLEANUP:
        shutil.rmtree(OZ / "sample_project")


@pytest.fixture(scope="session")
def base_project_directory(local_path) -> Path:
    return local_path / "sample_project"


@pytest.fixture(scope="session")
def base_overlay_directory(local_path) -> Path:
    return local_path / "sample_overlay"


@pytest.fixture(scope="session")
def base_config_overlay_only(local_path) -> Path:
    return local_path / "filesystem_overlay_only.json"


@pytest.fixture(scope="session")
def base_config(local_path) -> Path:
    return local_path / "filesystem.json"


COMMON_CASES = []


@pytest.fixture
def local_overlay_only(base_project_directory, base_config_overlay_only):
    # Copy project to temp directory
    temp_dir = Path(tempfile.mkdtemp())
    project_directory = temp_dir / "sample_project_overlay"
    config = temp_dir / "local_overlay_only.json"
    shutil.copytree(base_project_directory, project_directory)

    # Open baseline config
    with open(base_config_overlay_only, "r") as f:
        cfg = json.load(f)

    # Set the overlay root to the sample project
    cfg["overlay_root"] = "local://" + str(project_directory)
    cfg["overlay_fs_args"] = {"auto_mkdir": True}

    # Write the config to the local path
    with open(config, "w") as f:
        json.dump(cfg, f)

    payload = {
        "cfg_file": config,
        "testfs_static": None,
        "testpath_static": None,
        "testfs_overlay": fsspec.filesystem("local"),
        "testpath_overlay": PurePath(project_directory),
    }

    yield payload

    if CLEANUP:
        shutil.rmtree(temp_dir)


@pytest.fixture
def local(base_project_directory, base_overlay_directory, base_config):
    # Copy project to temp directory
    temp_dir = Path(tempfile.mkdtemp())
    project_directory = temp_dir / "sample_project"
    overlay_directory = temp_dir / "sample_overlay"
    config = temp_dir / "local.json"
    shutil.copytree(base_project_directory, project_directory)
    shutil.copytree(base_overlay_directory, overlay_directory)

    with open(base_config, "r") as f:
        cfg = json.load(f)

    # Set the overlay root to the sample project overlay
    cfg["overlay_root"] = "local://" + str(overlay_directory)
    cfg["overlay_fs_args"] = {"auto_mkdir": True}

    # Set the static root to the sample project
    cfg["static_root"] = "local://" + str(project_directory)
    cfg["static_fs_args"] = {"auto_mkdir": False}

    # Write the config to the local path
    with open(config, "w") as f:
        json.dump(cfg, f)

    payload = {
        "cfg_file": config,
        "testfs_static": fsspec.filesystem("local"),
        "testpath_static": PurePath(project_directory),
        "testfs_overlay": fsspec.filesystem("local"),
        "testpath_overlay": PurePath(overlay_directory),
    }

    yield payload

    if CLEANUP:
        shutil.rmtree(temp_dir)


COMMON_CASES.extend(["local_overlay_only", "local"])


if importlib_util.find_spec("s3fs") and RUN_ALL:

    @pytest.fixture(scope="session")
    def s3_container():
        os.system("docker compose -f ./tests/docker-compose.yml --profile s3fs up -d")
        yield "s3://test-bucket/"
        os.system("docker compose -f ./tests/docker-compose.yml --profile '*' stop")

    @pytest.fixture
    def s3_overlay_only(s3_container, base_project_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "s3_overlay_only.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{s3_container}sample_project_overlay_only_{uuid.uuid1()}/"
        os.system(f'bash ./tests/seed_moto.sh "{base_project_directory}" "{project_directory}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(project_directory)
        cfg["overlay_fs_args"] = {
            "key": "test",
            "secret": "test",
            "endpoint_url": "http://127.0.0.1:4001",
            "client_kwargs": {"region_name": "us-west-2"},
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f)

        payload = {
            "cfg_file": config,
            "testfs_static": None,
            "testpath_static": None,
            "testfs_overlay": fsspec.filesystem("s3", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(project_directory.replace("s3://", "")),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def s3(s3_container, base_project_directory, base_overlay_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "s3.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{s3_container}sample_project_{uuid.uuid1()}/"
        os.system(f'bash ./tests/seed_moto.sh "{base_project_directory}" "{project_directory}"')

        overlay_directory = f"{s3_container}sample_overlay_{uuid.uuid1()}/"
        os.system(f'bash ./tests/seed_moto.sh "{base_overlay_directory}" "{overlay_directory}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(overlay_directory)
        cfg["overlay_fs_args"] = {
            "key": "test",
            "secret": "test",
            "endpoint_url": "http://127.0.0.1:4001",
            "client_kwargs": {"region_name": "us-west-2"},
        }

        # Set the overlay root to the sample project
        cfg["static_root"] = str(project_directory)
        cfg["static_fs_args"] = {
            "key": "test",
            "secret": "test",
            "endpoint_url": "http://127.0.0.1:4001",
            "client_kwargs": {"region_name": "us-west-2"},
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f)

        payload = {
            "cfg_file": config,
            "testfs_static": fsspec.filesystem("s3", **cfg["static_fs_args"]),
            "testpath_static": PurePath(project_directory.replace("s3://", "")),
            "testfs_overlay": fsspec.filesystem("s3", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(overlay_directory.replace("s3://", "")),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)

    COMMON_CASES.extend(["s3_overlay_only", "s3"])


if importlib_util.find_spec("sshfs") and RUN_ALL:

    @pytest.fixture(scope="session")
    def ssh_container():
        os.system("docker compose -f ./tests/docker-compose.yml --profile sshfs up -d")
        # On startup we need to wait a second for the service to start, otherwise the first test fails.
        # Let's not talk about how long it took to figure this out.
        time.sleep(1)
        yield "ssh:///tmp/"
        os.system("docker compose -f ./tests/docker-compose.yml --profile '*' stop")
        os.system("docker compose -f ./tests/docker-compose.yml --profile '*' rm -f")

    @pytest.fixture
    def ssh_overlay_only(ssh_container, base_project_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "ssh_overlay_only.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{ssh_container}sample_project_{uuid.uuid1()}"
        project_directory_stripped = project_directory.replace("ssh://", "")
        os.system(f'bash ./tests/seed_ssh.sh "{base_project_directory}/*" "{project_directory_stripped}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(project_directory)
        cfg["overlay_fs_args"] = {
            "host": "localhost",
            "port": 2222,
            "username": "test.user",
            "password": "password",
            "known_hosts": None,
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f, indent=4)

        payload = {
            "cfg_file": config,
            "testfs_static": None,
            "testpath_static": None,
            "testfs_overlay": fsspec.filesystem("ssh", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(project_directory_stripped),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def ssh(ssh_container, base_project_directory, base_overlay_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "ssh.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{ssh_container}sample_project_{uuid.uuid1()}"
        project_directory_stripped = project_directory.replace("ssh://", "")
        os.system(f'bash ./tests/seed_ssh.sh "{base_project_directory}/*" "{project_directory_stripped}"')

        overlay_directory = f"{ssh_container}sample_overlay_{uuid.uuid1()}"
        overlay_directory_stripped = overlay_directory.replace("ssh://", "")
        os.system(f'bash ./tests/seed_ssh.sh "{base_overlay_directory}/*" "{overlay_directory_stripped}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(overlay_directory)
        cfg["overlay_fs_args"] = {
            "host": "localhost",
            "port": 2222,
            "username": "test.user",
            "password": "password",
            "known_hosts": None,
        }

        # Set the overlay root to the sample project
        cfg["static_root"] = str(project_directory)
        cfg["static_fs_args"] = {
            "host": "localhost",
            "port": 2222,
            "username": "test.user",
            "password": "password",
            "known_hosts": None,
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f, indent=4)

        payload = {
            "cfg_file": config,
            "testfs_static": fsspec.filesystem("ssh", **cfg["static_fs_args"]),
            "testpath_static": PurePath(project_directory_stripped),
            "testfs_overlay": fsspec.filesystem("ssh", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(overlay_directory_stripped),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)

    COMMON_CASES.extend(["ssh_overlay_only", "ssh"])


if importlib_util.find_spec("smbclient") and RUN_ALL:

    @pytest.fixture(scope="session")
    def smb_container():
        os.system("docker compose -f ./tests/docker-compose.yml --profile smb up -d")
        # On startup we need to wait a second for the service to start, otherwise the first test fails.
        time.sleep(2)
        yield "smb:///data/"
        os.system("docker compose -f ./tests/docker-compose.yml --profile '*' stop")

    @pytest.fixture
    def smb_overlay_only(smb_container, base_project_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "smb_overlay_only.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{smb_container}sample_project_{uuid.uuid1()}"
        project_directory_stripped = project_directory.replace("smb:///data/", "")
        os.system(f'bash ./tests/seed_smb.sh "{base_project_directory}/*" "{project_directory_stripped}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(project_directory)
        cfg["overlay_fs_args"] = {
            "host": "localhost",
            "username": "test.user",
            "password": "password",
            "auto_mkdir": True,
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f, indent=4)

        payload = {
            "cfg_file": config,
            "testfs_static": None,
            "testpath_static": None,
            "testfs_overlay": fsspec.filesystem("smb", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(project_directory.replace("smb://", "")),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)
            shutil.rmtree(f"tests/bin/smb/{project_directory_stripped}")

    @pytest.fixture
    def smb(smb_container, base_project_directory, base_overlay_directory, base_config_overlay_only):
        # Temp dir for config
        temp_dir = Path(tempfile.mkdtemp())
        config = temp_dir / "smb.json"

        # To ensure that each test has a unique project directory, generate UUID names
        project_directory = f"{smb_container}sample_project_{uuid.uuid1()}"
        project_directory_stripped = project_directory.replace("smb:///data/", "")
        os.system(f'bash ./tests/seed_smb.sh "{base_project_directory}/*" "{project_directory_stripped}"')

        overlay_directory = f"{smb_container}sample_overlay_{uuid.uuid1()}"
        overlay_directory_stripped = overlay_directory.replace("smb:///data/", "")
        os.system(f'bash ./tests/seed_smb.sh "{base_overlay_directory}/*" "{overlay_directory_stripped}"')

        # Open baseline config
        with open(base_config_overlay_only, "r") as f:
            cfg = json.load(f)

        # Set the overlay root to the sample project
        cfg["overlay_root"] = str(overlay_directory)
        cfg["overlay_fs_args"] = {
            "host": "localhost",
            "username": "test.user",
            "password": "password",
            "auto_mkdir": True,
        }

        # Set the overlay root to the sample project
        cfg["static_root"] = str(project_directory)
        cfg["static_fs_args"] = {
            "host": "localhost",
            "username": "test.user",
            "password": "password",
        }

        # Write the config to the local path
        with open(config, "w") as f:
            json.dump(cfg, f, indent=4)

        payload = {
            "cfg_file": config,
            "testfs_static": fsspec.filesystem("smb", **cfg["static_fs_args"]),
            "testpath_static": PurePath(project_directory.replace("smb://", "")),
            "testfs_overlay": fsspec.filesystem("smb", **cfg["overlay_fs_args"]),
            "testpath_overlay": PurePath(overlay_directory.replace("smb://", "")),
        }

        yield payload

        if CLEANUP:
            shutil.rmtree(temp_dir)
            shutil.rmtree(f"tests/bin/smb/{project_directory_stripped}")
            shutil.rmtree(f"tests/bin/smb/{overlay_directory_stripped}")

    COMMON_CASES.extend(["smb_overlay_only", "smb"])


def pytest_configure():
    pytest.common_cases = COMMON_CASES
