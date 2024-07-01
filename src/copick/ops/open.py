import json
import warnings

from copick.impl.cryoet_data_portal import CopickConfigCDP, CopickRootCDP
from copick.impl.filesystem import CopickConfigFSSpec, CopickRootFSSpec


def from_file(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    if "config_type" not in data:
        data["config_type"] = "filesystem"
        warnings.warn(
            "config_type not found in config file, defaulting to filesystem",
            DeprecationWarning,
            stacklevel=2,
        )

    if data["config_type"] == "filesystem":
        return CopickRootFSSpec(CopickConfigFSSpec(**data))
    elif data["config_type"] == "cryoet_data_portal":
        return CopickRootCDP(CopickConfigCDP(**data))
