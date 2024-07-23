from typing import Any, Dict

import numpy as np
import pytest
import zarr
from copick.impl.filesystem import CopickRootFSSpec
from copick.models import CopickPicksFile
from copick.util.ome import write_ome_zarr_3d
from trimesh.parent import Geometry

NUMERICAL_PRECISION = 1e-8


@pytest.fixture(params=pytest.common_cases)
def test_payload(request) -> Dict[str, Any]:
    payload = request.getfixturevalue(request.param)
    payload["root"] = CopickRootFSSpec.from_file(payload["cfg_file"])
    return payload


def test_root_lazy(test_payload: Dict[str, Any]):
    copick_root = test_payload["root"]

    # Check that the root is not populated
    assert copick_root._runs is None, "Runs should not be populated"
    assert copick_root._objects is None, "Objects should not be populated"

    # Access the runs and confirm query
    runs = copick_root.runs
    rnames = [r.name for r in runs]

    assert copick_root._runs is not None, "Runs should be populated"
    assert len(runs) == 3, "Incorrect number of runs"
    assert set(rnames) == {"TS_001", "TS_002", "TS_003"}, "Incorrect runs"

    # Access the objects and confirm query
    objects = copick_root.pickable_objects
    onames = [o.name for o in objects]

    assert copick_root._objects is not None, "Objects should be populated"
    assert len(objects) == 3, "Incorrect number of objects"
    assert set(onames) == {"proteasome", "ribosome", "membrane"}, "Incorrect objects"


def test_root_metadata(test_payload: Dict[str, Any]):
    copick_root = test_payload["root"]

    assert copick_root.config.name == "test", "Incorrect name"
    assert copick_root.config.description == "A test project.", "Incorrect description"

    assert copick_root.user_id is None, "Incorrect user_id"
    assert copick_root.session_id is None, "Incorrect description"

    copick_root.user_id = "test.user"
    copick_root.session_id = "0"

    assert copick_root.user_id == "test.user", "Incorrect user_id after setting"
    assert copick_root.session_id == "0", "Incorrect session_id after setting"


def test_root_get_object(test_payload: Dict[str, Any]):
    copick_root = test_payload["root"]

    objs = ["proteasome", "ribosome", "membrane"]

    for obj in objs:
        assert copick_root.get_object(obj) is not None, f"Object {obj} not found"
        assert copick_root.get_object(obj).name == obj, f"Object {obj} not found"

    assert copick_root.get_object("nucleus") is None, "Object nucleus should not be found"


def test_root_get_run(test_payload: Dict[str, Any]):
    copick_root = test_payload["root"]

    runs = ["TS_001", "TS_002", "TS_003"]

    for run in runs:
        assert copick_root.get_run(run) is not None, f"Run {run} not found"
        assert copick_root.get_run(run).name == run, f"Run {run} not found"
        assert copick_root._runs is None, f"Random access for run {run} should not populate runs index"

    assert copick_root.get_run("TS_004") is None, "Run TS_004 should not be found"


def test_root_refresh(test_payload: Dict[str, Any]):
    copick_root = test_payload["root"]

    # Check that the root is not populated
    assert copick_root._runs is None, "Runs should not be populated"

    copick_root.refresh()

    assert copick_root._runs is not None, "Runs should be populated"
    rnames = [r.name for r in copick_root.runs]
    assert len(copick_root.runs) == 3, "Incorrect number of runs"
    assert set(rnames) == {"TS_001", "TS_002", "TS_003"}, "Incorrect runs"


def test_root_new_run(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that the root is not populated
    assert copick_root._runs is None, "Runs should not be populated"

    # Adding a run with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        copick_root.new_run("TS_001")

    # Adding the first run inits the _runs attribute as list of runs
    run4 = copick_root.new_run("TS_004")
    assert copick_root._runs is not None, "Runs should be populated"
    assert run4 in copick_root.runs, "Run not added to runs"

    # Adding another run appends to the list
    run5 = copick_root.new_run("TS_005")
    assert run5 in copick_root.runs, "Run not added to runs"

    # Total number of runs should be 5 now
    copick_root.refresh()
    assert len(copick_root.runs) == 5, "Incorrect number of runs"

    # Check filesystem for existing runs
    if only_overlay:
        run_path = str(overlay_loc / "ExperimentRuns") + "/"
        for run in ["TS_001", "TS_002", "TS_003", "TS_004", "TS_005"]:
            assert overlay_fs.exists(run_path + run), f"{run} not found in overlay"
    else:
        run_path_overlay = str(overlay_loc / "ExperimentRuns") + "/"
        run_path_static = str(static_loc / "ExperimentRuns") + "/"

        for run in ["TS_001", "TS_002", "TS_003"]:
            assert static_fs.exists(run_path_static + run), f"{run} not found in static"

        for run in ["TS_004", "TS_005"]:
            assert overlay_fs.exists(run_path_overlay + run), f"{run} not found in overlay"


def test_object_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Check metadata for proteasome object
    copick_object = copick_root.get_object("proteasome")
    assert copick_object.name == "proteasome", "Incorrect name"
    assert copick_object.is_particle is True, "Incorrect is_particle"
    assert copick_object.label == 1, "Incorrect label"
    assert copick_object.color == (255, 0, 0, 255), "Incorrect color"
    assert copick_object.radius == 60, "Incorrect radius"
    assert copick_object.map_threshold == pytest.approx(0.0418, abs=NUMERICAL_PRECISION), "Incorrect threshold"


def test_object_zarr(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Check metadata for proteasome object
    copick_object = copick_root.get_object("proteasome")

    # Check zarr is readable
    arrays = list(zarr.open(copick_object.zarr(), "r").arrays())
    _, array = arrays[0]
    assert array.shape == (42, 36, 36), "Error reading Zarr, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        1029.290283203125,
        abs=NUMERICAL_PRECISION,
    ), "Error reading Zarr (incorrect sum)."

    # Assert no zarr for non-particle object
    copick_object = copick_root.get_object("membrane")
    assert copick_object.zarr() is None, "Zarr should not exist for non-particle object"


def test_object_read_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Check metadata for proteasome object
    copick_object = copick_root.get_object("proteasome")

    # Check numpy is readable
    array = copick_object.numpy()

    # Full size
    assert array.shape == (42, 36, 36), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        1029.290283203125,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."

    # Subregion
    array = copick_object.numpy(x=slice(10, 20), y=slice(10, 20), z=slice(10, 20))
    assert array.shape == (10, 10, 10), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        168.116912842,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."


def test_run_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Check metadata for TS_001 run
    copick_run = copick_root.get_run("TS_001")

    # Name
    assert copick_run.name == "TS_001", "Incorrect name"
    copick_run.name = "TS_001_new"
    assert copick_run.name == "TS_001_new", "Incorrect name after setting"


def test_run_lazy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Check that the run is not populated
    copick_run = copick_root.get_run("TS_001")
    assert copick_run._voxel_spacings is None, "Voxel spacings should not be populated"
    assert copick_run._picks is None, "Picks should not be populated"
    assert copick_run._meshes is None, "Meshes should not be populated"
    assert copick_run._segmentations is None, "Segmentations should not be populated"

    # Access the voxel spacings and confirm query
    voxel_spacings = copick_run.voxel_spacings
    vs = [v.voxel_size for v in voxel_spacings]
    assert copick_run._voxel_spacings is not None, "Voxel spacings should be populated"
    assert len(vs) == 2, "Incorrect number of voxel spacings"
    assert set(vs) == {10.000, 20.000}, "Incorrect voxel spacings"

    # Access the picks and confirm query
    picks = copick_run.picks

    assert copick_run._picks is not None, "Picks should be populated"
    assert len(picks) == 5, "Incorrect number of picks"

    # Access the meshes and confirm query
    meshes = copick_run.meshes

    assert copick_run._meshes is not None, "Meshes should be populated"
    assert len(meshes) == 3, "Incorrect number of meshes"

    # Access the segmentations and confirm query
    segmentations = copick_run.segmentations

    assert copick_run._segmentations is not None, "Segmentations should be populated"
    assert len(segmentations) == 3, "Incorrect segmentations"


def test_run_get_voxel_spacing(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Non-existing voxel spacing
    assert copick_run.get_voxel_spacing(30.000) is None, "Voxel spacing should not exist"

    # Get voxel spacing
    vs = copick_run.get_voxel_spacing(10.000)
    assert vs is not None, "Voxel spacing not found"
    assert vs.voxel_size == 10.000, "Incorrect voxel size"
    assert (
        copick_run._voxel_spacings is None
    ), "Random access for voxel spacing should not populate voxel spacings index"

    vs = copick_run.get_voxel_spacing(20.000)
    assert vs is not None, "Voxel spacing not found"
    assert vs.voxel_size == 20.000, "Incorrect voxel size"


def test_run_entity_types(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Check pick types
    tool_picks = copick_run.tool_picks()
    assert len(tool_picks) == 2, "Incorrect number of tool picks"

    user_picks = copick_run.user_picks()
    assert len(user_picks) == 3, "Incorrect number of user picks"

    # Check mesh types
    tool_meshes = copick_run.tool_meshes()
    assert len(tool_meshes) == 2, "Incorrect number of tool meshes"

    user_meshes = copick_run.user_meshes()
    assert len(user_meshes) == 1, "Incorrect number of user meshes"

    # Check segmentation types
    tool_segmentations = copick_run.tool_segmentations()
    assert len(tool_segmentations) == 2, "Incorrect number of tool segmentations"

    user_segmentations = copick_run.user_segmentations()
    assert len(user_segmentations) == 1, "Incorrect number of user segmentations"

    # Check with user_id set
    copick_root.config.user_id = "test.user"
    user_picks = copick_run.user_picks()
    assert len(user_picks) == 2, "Incorrect number of user picks"

    user_meshes = copick_run.user_meshes()
    assert len(user_meshes) == 0, "Incorrect number of user meshes"

    user_segmentations = copick_run.user_segmentations()
    assert len(user_segmentations) == 1, "Incorrect number of user segmentations"


def test_run_get_picks(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get picks
    picks = copick_run.get_picks()
    assert len(picks) == 5, "Incorrect number of picks"

    # Get picks by object
    picks = copick_run.get_picks(object_name="proteasome")
    assert len(picks) == 3, "Incorrect number of picks"

    # Get picks by object and user_id
    picks = copick_run.get_picks(object_name="proteasome", user_id="ArtiaX")
    assert len(picks) == 1, "Incorrect number of picks"
    assert picks[0].pickable_object_name == "proteasome", "Incorrect object"

    # Get picks by object, user_id and session_id
    picks = copick_run.get_picks(object_name="ribosome", user_id="gapstop", session_id="0")
    assert len(picks) == 1, "Incorrect number of picks"
    assert picks[0].pickable_object_name == "ribosome", "Incorrect object"
    assert picks[0].user_id == "gapstop", "Incorrect user_id"
    assert picks[0].session_id == "0", "Incorrect session_id"


def test_picks_read_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get picks by object, user_id and session_id
    picks = copick_run.get_picks(object_name="ribosome", user_id="gapstop", session_id="0")

    POINTS = np.array(
        [
            [13.109512186388912, 74.71185201921038, 202.47507943202584],
            [631.4709314294199, 216.35459177569777, 153.51979459358296],
            [13.269106483949642, 493.1507625056122, 191.96928603555477],
            [295.3857864878723, 48.00127558957804, 12.379751870875282],
            [204.07695344548807, 363.70538811345045, 421.5631049171766],
            [341.5641484136662, 583.1178654724652, 265.92234830389776],
            [595.6901903739324, 152.54150479358174, 269.050095901217],
            [414.749742830462, 33.87702343343186, 149.04501288274474],
            [230.77112217392795, 158.07881774104047, 33.889195589406285],
            [205.03836780107628, 191.17304087417068, 17.748902885347206],
        ],
    )

    ORIENTATIONS = np.array(
        [
            [
                [0.1802696888767692, 0.019475241487624584, 0.4632185264983446, 0.0],
                [0.42020360458772743, 0.4854270981677824, 0.012780814590608647, 0.0],
                [0.9418066523433661, 0.8507950893767787, 0.7299644702208186, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.7589795881605054, 0.8182753578641435, 0.34462449095921743, 0.0],
                [0.11166123199466038, 0.08395314332205706, 0.7127259356888429, 0.0],
                [0.055673679585078406, 0.47979728165943036, 0.4016764806306522, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.06368643077465541, 0.3646156429960411, 0.07002280262450244, 0.0],
                [0.07038259510695688, 0.29026366713531204, 0.7901011234337895, 0.0],
                [0.7926213850799262, 0.5618187101448784, 0.6160183850726514, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.7346677209045017, 0.9329046271873459, 0.4003284329218084, 0.0],
                [0.8065667909277211, 0.7644910965444714, 0.6526147399155839, 0.0],
                [0.6422148587410551, 0.9574440528259551, 0.33387442145109236, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.4527591644919976, 0.6228340076529896, 0.7399515177978947, 0.0],
                [0.36761762506764994, 0.16902881059075237, 0.7938724066348071, 0.0],
                [0.734212987305353, 0.8752945800236829, 0.6480564419758135, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.31818719398704765, 0.21800337527051994, 0.8449385308721757, 0.0],
                [0.43822067087734373, 0.09141458692565019, 0.30097916984093387, 0.0],
                [0.34371330696030433, 0.7093010177714357, 0.9739682557652591, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.5190077415876589, 0.9224502382708699, 0.723911262619614, 0.0],
                [0.4974963527482624, 0.6487221982847005, 0.4477711603657113, 0.0],
                [0.9711752036309947, 0.9324137189386374, 0.8406854303886448, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.5954866904630542, 0.9321486216692217, 0.5388502918850757, 0.0],
                [0.7129629202330348, 0.5215734913958031, 0.505473168619571, 0.0],
                [0.14491946190451044, 0.9296051143502445, 0.9560582640482871, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.055084832105785964, 0.8464969771473587, 0.9890811615793559, 0.0],
                [0.05411530447521384, 0.9796933083190055, 0.7672162846141741, 0.0],
                [0.47825664023283687, 0.9155030040986599, 0.35583007986260196, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.07837980771565989, 0.9521754802942424, 0.7061476278277338, 0.0],
                [0.8144408709150506, 0.19072161285288602, 0.1759034817259575, 0.0],
                [0.29103797580332835, 0.7906135259730082, 0.17079014782816504, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
    )

    pos, ori = picks[0].numpy()
    assert np.allclose(pos, POINTS, atol=NUMERICAL_PRECISION), "Error getting numpy array (incorrect positions)."
    assert np.allclose(
        ori,
        ORIENTATIONS,
        atol=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect orientations)."


def test_picks_write_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get picks by object, user_id and session_id
    picks = copick_run.new_picks(object_name="ribosome", user_id="gapstop", session_id="1")

    POINTS = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )

    POINTS_err = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    ORIENTATIONS = np.array(
        [
            [
                [0.055084832105785964, 0.8464969771473587, 0.9890811615793559, 0.0],
                [0.05411530447521384, 0.9796933083190055, 0.7672162846141741, 0.0],
                [0.47825664023283687, 0.9155030040986599, 0.35583007986260196, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.07837980771565989, 0.9521754802942424, 0.7061476278277338, 0.0],
                [0.8144408709150506, 0.19072161285288602, 0.1759034817259575, 0.0],
                [0.29103797580332835, 0.7906135259730082, 0.17079014782816504, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
    )

    # Write picks
    picks.from_numpy(POINTS, ORIENTATIONS)
    del picks

    # Compare with file contents
    picks = copick_run.get_picks(object_name="ribosome", user_id="gapstop", session_id="1")[0]

    assert len(picks.points) == 2, "Incorrect number of points."

    for i, p in enumerate(picks.points):
        assert p.transformation == pytest.approx(
            ORIENTATIONS[i, :, :],
            abs=NUMERICAL_PRECISION,
        ), f"Incorrect position for point {i}."

        assert [p.location.x, p.location.y, p.location.z] == pytest.approx(
            POINTS[i, :],
            abs=NUMERICAL_PRECISION,
        ), f"Incorrect orientation for point {i}."

    with pytest.raises(ValueError):
        picks.from_numpy(POINTS_err, ORIENTATIONS)


def test_run_get_meshes(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get meshes
    meshes = copick_run.get_meshes()
    assert len(meshes) == 3, "Incorrect number of meshes"

    # Get meshes by object
    meshes = copick_run.get_meshes(object_name="proteasome")
    assert len(meshes) == 1, "Incorrect number of meshes"
    assert meshes[0].pickable_object_name == "proteasome", "Incorrect mesh"

    # Get meshes by user_id
    meshes = copick_run.get_meshes(user_id="user.test")
    assert len(meshes) == 1, "Incorrect number of meshes"
    assert meshes[0].user_id == "user.test", "Incorrect mesh"

    # Get meshes by session_id
    meshes = copick_run.get_meshes(session_id="321")
    assert len(meshes) == 1, "Incorrect number of meshes"
    assert meshes[0].session_id == "321", "Incorrect mesh"


def test_run_get_segmentations(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get segmentations
    segmentations = copick_run.get_segmentations()
    assert len(segmentations) == 3, "Incorrect number of segmentations"

    # Get segmentations by voxel spacing
    segmentations = copick_run.get_segmentations(voxel_size=10.000)
    snames = [s.name for s in segmentations]
    assert len(segmentations) == 2, "Incorrect number of segmentations"
    assert "prediction" in snames, "Segmentation not found"
    assert "painting" in snames, "Segmentation not found"

    # Get segmentations by object
    segmentations = copick_run.get_segmentations(name="membrane")
    assert len(segmentations) == 1, "Incorrect number of segmentations"
    assert segmentations[0].name == "membrane", "Incorrect segmentation"

    # Get segmentations by user_id
    segmentations = copick_run.get_segmentations(user_id="membrain")
    assert len(segmentations) == 1, "Incorrect number of segmentations"
    assert segmentations[0].user_id == "membrain", "Incorrect segmentation"

    # Get segmentations by multilabel and session_id
    segmentations = copick_run.get_segmentations(is_multilabel=True, session_id="123")
    assert len(segmentations) == 1, "Incorrect number of segmentations"
    assert segmentations[0].is_multilabel is True, "Incorrect segmentation"
    assert segmentations[0].session_id == "123", "Incorrect segmentation"


def test_segmentation_zarr(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get segmentations by object
    segmentation = copick_run.get_segmentations(name="membrane")[0]

    # Check zarr is readable
    arrays = list(zarr.open(segmentation.zarr(), "r").arrays())
    _, array = arrays[0]
    assert array.shape == (64, 64, 64), "Error reading Zarr, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        24576,
        abs=NUMERICAL_PRECISION,
    ), "Error reading Zarr (incorrect sum)."


def test_segmentation_read_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get segmentations by object
    segmentation = copick_run.get_segmentations(name="membrane")[0]

    # Full volume
    array = segmentation.numpy()
    assert array.shape == (64, 64, 64), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        24576,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."

    # Subregion
    array = segmentation.numpy(x=slice(20, 40), y=slice(20, 40), z=slice(20, 40))
    assert array.shape == (20, 20, 20), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        426,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."


def test_segmentation_write_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]

    # Get run
    copick_run = copick_root.get_run("TS_001")

    # Get new segmentation
    segmentation = copick_run.new_segmentation(
        name="ribosome",
        user_id="pytom",
        session_id="10",
        is_multilabel=False,
        voxel_size=10.000,
    )

    # Write numpy array
    array = np.random.randint(low=0, high=50, size=(64, 64, 64)).astype(np.uint8)
    segmentation.from_numpy(array)

    # Check zarr contents
    arrays = list(zarr.open(segmentation.zarr(), "r").arrays())
    _, array2 = arrays[0]
    assert np.allclose(array, array2), "Error writing numpy array"

    # Write subregion
    sub_array = np.random.rand(30, 30, 30).astype(np.uint8)
    franken_array = array
    franken_array[10:40, 10:40, 10:40] = sub_array
    segmentation.set_region(sub_array, x=slice(10, 40), y=slice(10, 40), z=slice(10, 40))

    # Check zarr contents
    arrays = list(zarr.open(segmentation.zarr(), "r").arrays())
    _, array2 = arrays[0]
    assert np.allclose(franken_array, array2), "Error writing numpy array subregion"


def test_run_new_voxel_spacing(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")

    # Adding a voxel spacing with the same size as an existing one raises an error
    with pytest.raises(ValueError):
        copick_run.new_voxel_spacing(10.000)

    # Adding the first voxel spacing inits the _voxel_spacings attribute as list of voxel spacings
    vs3 = copick_run.new_voxel_spacing(30.000)

    assert copick_run._voxel_spacings is not None, "Voxel spacings should be populated"
    assert vs3 in copick_run.voxel_spacings, "Voxel spacing not added to voxel spacings"

    # Adding another voxel spacing appends to the list
    vs4 = copick_run.new_voxel_spacing(40.000)
    assert vs4 in copick_run.voxel_spacings, "Voxel spacing not added to voxel spacings"

    # Total number of voxel spacings should be 4 now
    copick_run.refresh_voxel_spacings()
    assert len(copick_run.voxel_spacings) == 4, "Incorrect number of voxel spacings"


def test_run_new_picks(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")

    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that picks are not populated
    assert copick_run._picks is None, "Picks should not be populated"

    # Adding a pick with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        copick_run.new_picks(object_name="proteasome", session_id="0", user_id="pytom")

    # Add pick with object that does not exist
    with pytest.raises(ValueError):
        copick_run.new_picks(object_name="nucleus", session_id="0", user_id="pytom")

    # Adding the first pick inits the _picks attribute as list of picks
    pick6 = copick_run.new_picks(object_name="ribosome", session_id="0", user_id="ArtiaX")

    assert copick_run._picks is not None, "Picks should be populated"
    assert pick6 in copick_run.picks, "Pick not added to picks"

    # Adding another pick appends to the list after setting user id
    copick_root.config.user_id = "user.test"
    pick7 = copick_run.new_picks(object_name="ribosome", session_id="1234")

    assert pick7 in copick_run.picks, "Pick not added to picks"
    assert (
        pick7 == copick_run.get_picks(object_name="ribosome", session_id="1234", user_id="user.test")[0]
    ), "Pick not found"

    # Total number of picks should be 7 now
    copick_run.refresh_picks()
    assert len(copick_run.picks) == 7, "Incorrect number of picks"

    # Check filesystem for existing picks
    st = [
        "pytom_0_proteasome.json",
        "test.user_1234_ribosome.json",
        "gapstop_0_ribosome.json",
        "test.user_1234_proteasome.json",
        "ArtiaX_19_proteasome.json",
    ]

    ov = [
        "ArtiaX_0_ribosome.json",
        "user.test_1234_ribosome.json",
    ]
    if only_overlay:
        pick_path = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Picks") + "/"
        for pick in ov + st:
            assert overlay_fs.exists(pick_path + pick), f"{pick} not found in overlay"
    else:
        pick_path_overlay = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Picks") + "/"
        pick_path_static = str(static_loc / "ExperimentRuns" / "TS_001" / "Picks") + "/"

        for pick in st:
            assert static_fs.exists(pick_path_static + pick), f"{pick} not found in static"

        for pick in ov:
            assert overlay_fs.exists(pick_path_overlay + pick), f"{pick} not found in overlay"


def test_run_new_meshes(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")

    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that the meshes are not populated
    assert copick_run._meshes is None, "Meshes should not be populated"

    # Adding a mesh with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        copick_run.new_mesh(object_name="membrane", session_id="0", user_id="membrain")

    # Add mesh with object that does not exist
    with pytest.raises(ValueError):
        copick_run.new_mesh(object_name="nucleus", session_id="0", user_id="pytom")

    # Adding the first mesh inits the _meshes attribute as list of meshes
    mesh2 = copick_run.new_mesh(object_name="membrane", session_id="0", user_id="ArtiaX")

    assert copick_run._meshes is not None, "Meshes should be populated"
    assert mesh2 in copick_run.meshes, "Mesh not added to meshes"

    # Adding another mesh appends to the list after setting user id
    copick_root.config.user_id = "user.test"
    mesh3 = copick_run.new_mesh(object_name="membrane", session_id="1234")

    assert mesh3 in copick_run.meshes, "Mesh not added to meshes"
    assert (
        mesh3 == copick_run.get_meshes(object_name="membrane", session_id="1234", user_id="user.test")[0]
    ), "Mesh not found"

    # Total number of meshes should be 5 now
    copick_run.refresh_meshes()
    assert len(copick_run.meshes) == 5, "Incorrect number of meshes"

    # Check filesystem for existing meshes
    st = [
        "membrain_0_membrane.glb",
        "user.test_321_proteasome.glb",
        "gapstop_0_ribosome.glb",
    ]

    ov = [
        "ArtiaX_0_membrane.glb",
        "user.test_1234_membrane.glb",
    ]

    if only_overlay:
        mesh_path = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Meshes") + "/"
        for mesh in ov + st:
            assert overlay_fs.exists(mesh_path + mesh), f"{mesh} not found in overlay"
    else:
        mesh_path_overlay = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Meshes") + "/"
        mesh_path_static = str(static_loc / "ExperimentRuns" / "TS_001" / "Meshes") + "/"

        for mesh in st:
            assert static_fs.exists(mesh_path_static + mesh), f"{mesh} not found in static"

        for mesh in ov:
            assert overlay_fs.exists(mesh_path_overlay + mesh), f"{mesh} not found in overlay"


def test_run_new_segmentations(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")

    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that segmentations are not populated
    assert copick_run._segmentations is None, "Segmentations should not be populated"

    # Adding a segmentation with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        copick_run.new_segmentation(
            voxel_size=10.000,
            user_id="cellcanvas",
            session_id="0",
            name="prediction",
            is_multilabel=True,
        )

    # Add segmentation with object that does not exist
    with pytest.raises(ValueError):
        copick_run.new_segmentation(
            voxel_size=10.000,
            user_id="cellcanvas",
            session_id="0",
            name="nucleus",
            is_multilabel=False,
        )

    # Adding the first segmentation inits the _segmentations attribute as list of segmentations
    # For object stores we actually need to write to the zarr to create the "directory"
    seg4 = copick_run.new_segmentation(
        voxel_size=10.000,
        user_id="test.user",
        session_id="0",
        name="ribosome",
        is_multilabel=False,
    )
    zarr.create((5, 5, 5), store=seg4.zarr())
    assert copick_run._segmentations is not None, "Segmentations should be populated"
    assert seg4 in copick_run.segmentations, "Segmentation not added to segmentations"

    # Adding another segmentation appends to the list after setting user id
    # For object stores we actually need to write to the zarr to create the "directory"
    copick_root.config.user_id = "user.test"
    seg5 = copick_run.new_segmentation(
        voxel_size=10.000,
        session_id="1234",
        name="location",
        is_multilabel=True,
    )
    zarr.create((5, 5, 5), store=seg5.zarr())
    assert seg5 in copick_run.segmentations, "Segmentation not added to segmentations"
    assert (
        seg5
        == copick_run.get_segmentations(
            voxel_size=10.000,
            name="location",
            session_id="1234",
            user_id="user.test",
            is_multilabel=True,
        )[0]
    ), "Segmentation not found"

    # Total number of segmentations should be 5 now
    copick_run.refresh_segmentations()
    assert len(copick_run.segmentations) == 5, "Incorrect number of segmentations"

    # Check filesystem for existing segmentations
    st = [
        "10.000_cellcanvas_0_prediction-multilabel.zarr",
        "10.000_test.user_123_painting-multilabel.zarr",
        "20.000_membrain_0_membrane.zarr",
    ]

    ov = [
        "10.000_user.test_1234_location-multilabel.zarr",
        "10.000_test.user_0_ribosome.zarr",
    ]

    if only_overlay:
        seg_path = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Segmentations") + "/"
        for seg in ov + st:
            assert overlay_fs.exists(seg_path + seg), f"{seg} not found in overlay"
    else:
        seg_path_overlay = str(overlay_loc / "ExperimentRuns" / "TS_001" / "Segmentations") + "/"
        seg_path_static = str(static_loc / "ExperimentRuns" / "TS_001" / "Segmentations") + "/"

        for seg in st:
            assert static_fs.exists(seg_path_static + seg), f"{seg} not found in static"

        for seg in ov:
            assert overlay_fs.exists(seg_path_overlay + seg), f"{seg} not found in overlay"


def test_run_refresh(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")

    # Check that the run is not populated
    assert copick_run._voxel_spacings is None, "Voxel spacings should not be populated"
    assert copick_run._picks is None, "Picks should not be populated"
    assert copick_run._meshes is None, "Meshes should not be populated"
    assert copick_run._segmentations is None, "Segmentations should not be populated"

    copick_run.refresh()

    assert copick_run._voxel_spacings is not None, "Voxel spacings should be populated"
    assert copick_run._picks is not None, "Picks should be populated"
    assert copick_run._meshes is not None, "Meshes should be populated"
    assert copick_run._segmentations is not None, "Segmentations should be populated"

    assert len(copick_run.voxel_spacings) == 2, "Incorrect number of voxel spacings"
    assert len(copick_run.picks) == 5, "Incorrect number of picks"
    assert len(copick_run.meshes) == 3, "Incorrect number of meshes"
    assert len(copick_run.segmentations) == 3, "Incorrect number of segmentations"


def test_vs_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)

    # Check metadata for voxel spacing
    assert vs.voxel_size == 10.000, "Incorrect voxel size"


def test_vs_lazy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)

    # Check that the voxel spacing is not populated
    assert vs._tomograms is None, "Voxel size should not be populated"

    # Access the tomograms and confirm query
    tomograms = vs.tomograms
    ttype = [t.tomo_type for t in tomograms]

    assert len(tomograms) == 2, "Incorrect number of tomograms"
    assert "denoised" in ttype, "Expected tomogram not found"
    assert "wbp" in ttype, "Expected tomogram not found"


def test_vs_get_tomogram(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)

    # Get tomogram
    tomogram = vs.get_tomogram(tomo_type="denoised")
    assert tomogram is not None, "Tomogram not found"
    assert tomogram.tomo_type == "denoised", "Wrong tomogram found"

    # Non-existing tomogram
    tomogram = vs.get_tomogram(tomo_type="SIRT")
    assert tomogram is None, "Tomogram should not be found"


def test_vs_new_tomogram(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)

    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that the voxel spacing is not populated
    assert vs._tomograms is None, "Tomograms should not be populated"

    # Adding a tomogram with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        vs.new_tomogram(tomo_type="denoised")

    # Adding the first tomogram inits the _tomograms attribute as list of tomograms
    # For object stores we actually need to write to the zarr to create the "directory"
    tomogram = vs.new_tomogram(tomo_type="isonet")
    zarr.create((5, 5, 5), store=tomogram.zarr())

    assert vs._tomograms is not None, "Tomograms should be populated"
    assert tomogram in vs.tomograms, "Tomogram not added to tomograms"

    # Adding another tomogram appends to the list
    # For object stores we actually need to write to the zarr to create the "directory"
    tomogram = vs.new_tomogram(tomo_type="SIRT")
    zarr.create((5, 5, 5), store=tomogram.zarr())

    assert tomogram in vs.tomograms, "Tomogram not added to tomograms"
    assert tomogram == vs.get_tomogram(tomo_type="SIRT"), "Tomogram not found"

    # Total number of tomograms should be 3 now
    vs.refresh_tomograms()
    assert len(vs.tomograms) == 4, "Incorrect number of tomograms"

    # Check filesystem for existing tomograms
    st = [
        "denoised.zarr",
        "wbp.zarr",
    ]

    ov = [
        "isonet.zarr",
        "SIRT.zarr",
    ]

    if only_overlay:
        tomogram_path = str(overlay_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"
        for tomogram in ov + st:
            assert overlay_fs.exists(tomogram_path + tomogram), f"{tomogram} not found in overlay"
    else:
        tomogram_path_overlay = str(overlay_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"
        tomogram_path_static = str(static_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"

        for tomogram in st:
            assert static_fs.exists(tomogram_path_static + tomogram), f"{tomogram} not found in static"

        for tomogram in ov:
            assert overlay_fs.exists(tomogram_path_overlay + tomogram), f"{tomogram} not found in overlay"


def test_vs_refresh(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)

    # Check that the voxel spacing is not populated
    assert vs._tomograms is None, "Tomograms should not be populated"

    vs.refresh()

    assert vs._tomograms is not None, "Tomograms should be populated"
    assert len(vs.tomograms) == 2, "Incorrect number of tomograms"


def test_tomogram_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="denoised")

    # Check metadata for tomogram
    assert tomogram.tomo_type == "denoised", "Incorrect tomogram type"


def test_tomogram_lazy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")

    # Check that the features are not populated
    assert tomogram._features is None, "features should not be populated"

    # Access the features and confirm query
    features = tomogram.features
    ftype = [f.feature_type for f in features]

    assert len(features) == 2, "Incorrect number of features"
    assert "sobel" in ftype, "Expected feature not found"
    assert "edge" in ftype, "Expected feature not found"


def test_tomogram_get_features(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")

    # Get features by type
    features = tomogram.get_features(feature_type="sobel")
    assert features is not None, "Incorrect number of features"

    # Non-existing feature
    features = tomogram.get_features(feature_type="sift")
    assert features is None, "Incorrect number of features"


def test_tomogram_new_features(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")

    overlay_fs = test_payload["testfs_overlay"]
    overlay_loc = test_payload["testpath_overlay"]

    only_overlay = True
    static_fs = None
    static_loc = None
    if test_payload["testfs_static"] is not None:
        static_fs = test_payload["testfs_static"]
        static_loc = test_payload["testpath_static"]
        only_overlay = False

    # Check that the tomogram is not populated
    assert tomogram._features is None, "Features should not be populated"

    # Adding a feature with the same name as an existing one raises an error
    with pytest.raises(ValueError):
        tomogram.new_features(feature_type="sobel")

    # Adding the first feature inits the _features attribute as list of features
    # For object stores we actually need to write to the zarr to create the "directory"
    feature = tomogram.new_features(feature_type="sift")
    zarr.create((5, 5, 5), store=feature.zarr())

    assert tomogram._features is not None, "Features should be populated"
    assert feature in tomogram.features, "Feature not added to features"

    # Adding another feature appends to the list
    # For object stores we actually need to write to the zarr to create the "directory"
    feature = tomogram.new_features(feature_type="tomotwin")
    zarr.create((5, 5, 5), store=feature.zarr())

    assert feature in tomogram.features, "Feature not added to features"
    assert feature == tomogram.get_features(feature_type="tomotwin"), "Feature not found"

    # Total number of features should be 3 now
    tomogram.refresh_features()
    assert len(tomogram.features) == 4, "Incorrect number of features"

    # Check filesystem for existing features
    st = [
        "wbp_sobel_features.zarr",
        "wbp_edge_features.zarr",
    ]

    ov = [
        "wbp_sift_features.zarr",
        "wbp_tomotwin_features.zarr",
    ]

    if only_overlay:
        feature_path = str(overlay_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"
        for feature in ov + st:
            assert overlay_fs.exists(feature_path + feature), f"{feature} not found in overlay"
    else:
        feature_path_overlay = str(overlay_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"
        feature_path_static = str(static_loc / "ExperimentRuns" / "TS_001" / "VoxelSpacing10.000") + "/"

        for feature in st:
            assert static_fs.exists(feature_path_static + feature), f"{feature} not found in static"

        for feature in ov:
            assert overlay_fs.exists(feature_path_overlay + feature), f"{feature} not found in overlay"


def test_tomogram_refresh(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")

    # Check that the tomogram is not populated
    assert tomogram._features is None, "Features should not be populated"

    tomogram.refresh()

    assert tomogram._features is not None, "Features should be populated"
    assert len(tomogram.features) == 2, "Incorrect number of features"


def test_tomogram_zarr(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="denoised")

    # Check zarr is readable
    arrays = list(zarr.open(tomogram.zarr(), "r").arrays())
    _, array = arrays[0]
    assert array.shape == (64, 64, 64), "Error reading Zarr, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        8192.0,
        abs=NUMERICAL_PRECISION,
    ), "Error reading Zarr (incorrect sum)."

    # Check zarr is writable
    tomo = vs.new_tomogram(tomo_type="test")
    zarr.array(np.random.rand(64, 64, 64), store=tomo.zarr(), chunks=(32, 32, 32))


def test_tomogram_read_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="denoised")

    # Full array
    array = tomogram.numpy()
    assert array.shape == (64, 64, 64), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        8192.0,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."

    # Subregion
    array = tomogram.numpy(x=slice(0, 30), y=slice(50, 60), z=slice(10, 40))
    assert array.shape == (30, 10, 30), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        30.0,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."


def test_tomogram_write_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.new_tomogram(tomo_type="test")

    # Write numpy array
    array = np.random.rand(64, 64, 64)
    tomogram.from_numpy(array)

    # Check zarr contents
    arrays = list(zarr.open(tomogram.zarr(), "r").arrays())
    _, array2 = arrays[0]
    assert np.allclose(array, array2), "Error writing numpy array"

    # Write subregion
    sub_array = np.random.rand(30, 30, 30)
    franken_array = array
    franken_array[10:40, 10:40, 10:40] = sub_array
    tomogram.set_region(sub_array, x=slice(10, 40), y=slice(10, 40), z=slice(10, 40))

    # Check zarr contents
    arrays = list(zarr.open(tomogram.zarr(), "r").arrays())
    _, array2 = arrays[0]
    assert np.allclose(franken_array, array2), "Error writing numpy array subregion"


def test_feature_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")
    feature = tomogram.get_features(feature_type="sobel")

    # Check metadata for feature
    assert feature.feature_type == "sobel", "Incorrect feature type"
    assert feature.tomo_type == "wbp", "Incorrect feature name"


def test_feature_zarr(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")
    feature = tomogram.get_features(feature_type="sobel")

    # Check zarr is readable
    arrays = list(zarr.open(feature.zarr(), "r").arrays())
    _, array = arrays[0]
    assert array.shape == (64, 64, 64), "Error reading Zarr, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        20619.8125,
        abs=NUMERICAL_PRECISION,
    ), "Error reading Zarr (incorrect sum)."

    # Check zarr is writable
    feat = tomogram.new_features(feature_type="test")
    zarr.array(np.random.rand(64, 64, 64), store=feat.zarr(), chunks=(32, 32, 32))


def test_feature_read_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")
    feature = tomogram.get_features(feature_type="sobel")

    # Full volume
    array = feature.numpy()
    assert array.shape == (64, 64, 64), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        20619.8125,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."

    # Subregion
    array = feature.numpy(slices=(slice(20, 40), slice(20, 40), slice(20, 40)))
    assert array.shape == (20, 20, 20), "Error getting numpy array, (incorrect shape)"
    assert np.sum(array) == pytest.approx(
        563.36730957,
        abs=NUMERICAL_PRECISION,
    ), "Error getting numpy array (incorrect sum)."


def test_feature_write_numpy(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")
    feat = tomogram.new_features(feature_type="test")

    # Write zarr
    array = np.random.rand(64, 64, 64)
    pyramid = {10.000: array}
    write_ome_zarr_3d(feat.zarr(), pyramid, (32, 32, 32))

    # Write subregion
    sub_array = np.random.rand(30, 30, 30)
    franken_array = array
    franken_array[10:40, 10:40, 10:40] = sub_array
    feat.set_region(sub_array, slices=(slice(10, 40), slice(10, 40), slice(10, 40)))

    # Check zarr contents
    arrays = list(zarr.open(feat.zarr(), "r").arrays())
    _, array2 = arrays[0]
    assert np.allclose(franken_array, array2), "Error writing numpy array subregion"


def test_mesh_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    mesh = copick_run.get_meshes(object_name="membrane", session_id="0", user_id="membrain")[0]

    # Check metadata for mesh
    assert mesh.pickable_object_name == "membrane", "Incorrect object name"
    assert mesh.session_id == "0", "Incorrect session id"
    assert mesh.user_id == "membrain", "Incorrect user id"
    assert mesh.from_tool is True, "Incorrect from_tool"


def test_mesh_io(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    mesh = copick_run.get_meshes(object_name="membrane", session_id="0", user_id="membrain")[0]

    only_overlay = True
    if test_payload["testfs_static"] is not None:
        only_overlay = False

    # Mesh not initialized
    assert mesh._mesh is None, "Mesh should not be initialized"

    # Check mesh is readable
    msh = mesh.load()
    assert isinstance(msh, Geometry), "Error reading mesh"

    # Check mesh is initialized
    assert mesh._mesh is not None, "Mesh should be initialized"
    assert msh == mesh.mesh, "Mesh should be initialized"

    # Check static mesh not writable
    if not only_overlay:
        with pytest.raises(PermissionError):
            mesh.store()

    # Check mesh is writable
    msh2 = copick_run.new_mesh(object_name="proteasome", session_id="0", user_id="deepfinder")
    msh2.store()


def test_picks_meta(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    pick = copick_run.get_picks(object_name="proteasome", session_id="0", user_id="pytom")[0]

    # Check metadata for pick
    assert pick.pickable_object_name == "proteasome", "Incorrect object name"
    assert pick.session_id == "0", "Incorrect session id"
    assert pick.user_id == "pytom", "Incorrect user id"
    assert pick.from_tool is True, "Incorrect from_tool"
    assert pick.trust_orientation is True, "Incorrect trust_orientation"
    assert pick.color == (255, 0, 0, 255), "Incorrect color"


def test_pick_io(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    picks = copick_run.get_picks(object_name="proteasome", session_id="0", user_id="pytom")[0]

    only_overlay = True
    if test_payload["testfs_static"] is not None:
        only_overlay = False

    # Check picks is readable
    pck = picks.load()
    assert isinstance(pck, CopickPicksFile), "Error reading pick"

    # Check static picks not writable
    if not only_overlay:
        with pytest.raises(PermissionError):
            picks.store()

    # Check pick is writable
    pck2 = copick_run.new_picks(object_name="ribosome", session_id="0", user_id="pytom")
    pck2.store()


def test_repr(test_payload: Dict[str, Any]):
    # Setup
    copick_root = test_payload["root"]
    copick_run = copick_root.get_run("TS_001")
    vs = copick_run.get_voxel_spacing(10.000)
    tomogram = vs.get_tomogram(tomo_type="wbp")
    feature = tomogram.get_features(feature_type="sobel")
    mesh = copick_run.get_meshes(object_name="membrane", session_id="0", user_id="membrain")[0]
    pick = copick_run.get_picks(object_name="proteasome", session_id="0", user_id="pytom")[0]
    seg = copick_run.get_segmentations(name="membrane")[0]
    co = copick_root.pickable_objects[0]

    repr(copick_run)
    repr(vs)
    repr(tomogram)
    repr(feature)
    repr(mesh)
    repr(pick)
    repr(seg)
    repr(co)
