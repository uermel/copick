import json
from typing import Dict, List, Literal, MutableMapping, Optional, Tuple, Type, TypeVar, Union

import numpy as np

# Should work with pydantic 1 and 2
import pydantic
import trimesh

if pydantic.VERSION.startswith("1"):
    from pydantic import BaseModel, validator
elif pydantic.VERSION.startswith("2"):
    from pydantic.v1 import BaseModel, validator
else:
    raise ImportError(f"Unsupported pydantic version {pydantic.VERSION}.")

from trimesh.parent import Geometry

TPickableObject = TypeVar("TPickableObject", bound="PickableObject")
TCopickConfig = TypeVar("TCopickConfig", bound="CopickConfig")
TCopickLocation = TypeVar("TCopickLocation", bound="CopickLocation")
TCopickPoint = TypeVar("TCopickPoint", bound="CopickPoint")
TCopickPicks = TypeVar("TCopickPicks", bound="CopickPicks")
TCopickMesh = TypeVar("TCopickMesh", bound="CopickMesh")
TCopickSegmentation = TypeVar("TCopickSegmentation", bound="CopickSegmentation")
TCopickTomogram = TypeVar("TCopickTomogram", bound="CopickTomogram")
TCopickFeatures = TypeVar("TCopickFeatures", bound="CopickFeatures")
TCopickVoxelSpacing = TypeVar("TCopickVoxelSpacing", bound="CopickVoxelSpacing")
TCopickRun = TypeVar("TCopickRun", bound="CopickRun")
TCopickObject = TypeVar("TCopickObject", bound="CopickObject")
TCopickRoot = TypeVar("TCopickRoot", bound="CopickRoot")


class PickableObject(BaseModel):
    """Metadata for a pickable objects.

    Attributes:
        name: Name of the object.
        is_particle: Whether this object should be represented by points (True) or segmentation masks (False).
        label: Numeric label/id for the object, as used in multilabel segmentation masks. Must be unique.
        color: RGBA color for the object.
        emdb_id: EMDB ID for the object.
        pdb_id: PDB ID for the object.
        map_threshold: Threshold to apply to the map when rendering the isosurface.
        radius: Radius of the particle, when displaying as a sphere.
    """

    name: str
    is_particle: bool
    label: Optional[int]
    color: Optional[Tuple[int, int, int, int]]
    emdb_id: Optional[str] = None
    pdb_id: Optional[str] = None
    map_threshold: Optional[float] = None
    radius: Optional[float] = None

    @validator("label")
    def validate_label(cls, v) -> int:
        """Validate the label."""
        assert v != 0, "Label 0 is reserved for background."
        return v

    @validator("color")
    def validate_color(cls, v) -> Tuple[int, int, int, int]:
        """Validate the color."""
        assert len(v) == 4, "Color must be a 4-tuple (RGBA)."
        assert all(0 <= c <= 255 for c in v), "Color values must be in the range [0, 255]."
        return v


class CopickConfig(BaseModel):
    """Configuration for a copick project. Defines the available objects, user_id and optionally an index for runs.

    Attributes:
        name: Name of the CoPick project.
        description: Description of the CoPick project.
        version: Version of the CoPick API.
        pickable_objects (List[PickableObject]): Index for available pickable objects.
        user_id: Unique identifier for the user (e.g. when distributing the config file to users).
        session_id: Unique identifier for the session.
        runs: Index for run names.
        voxel_spacings: Index for available voxel spacings.
        tomograms: Index for available voxel spacings and tomogram types.
    """

    name: Optional[str] = "CoPick"
    description: Optional[str] = "Let's CoPick!"
    version: Optional[str] = "0.2.0"
    pickable_objects: List[TPickableObject]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    runs: Optional[List[str]] = None
    voxel_spacings: Optional[List[float]] = None
    tomograms: Optional[Dict[float, List[str]]] = {}

    @classmethod
    def from_file(cls, filename: str) -> TCopickConfig:
        """
        Load a CopickConfig from a file and create a CopickConfig object.

        Args:
            filename: path to the file

        Returns:
            CopickConfig: Initialized CopickConfig object

        """
        with open(filename) as f:
            return cls(**json.load(f))


class CopickLocation(BaseModel):
    """Location in 3D space.

    Attributes:
        x: x-coordinate.
        y: y-coordinate.
        z: z-coordinate.
    """

    x: float
    y: float
    z: float


class CopickPoint(BaseModel):
    """Point in 3D space with an associated orientation, score value and instance ID.

    Attributes:
        location (CopickLocation): Location in 3D space.
        transformation: Transformation matrix.
        instance_id: Instance ID.
        score: Score value.
    """

    location: TCopickLocation
    transformation_: Optional[List[List[float]]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    instance_id: Optional[int] = 0
    score: Optional[float] = 1.0

    class Config:
        arbitrary_types_allowed = True

    @validator("transformation_")
    def validate_transformation(cls, v) -> List[List[float]]:
        """Validate the transformation matrix."""
        arr = np.array(v)
        assert arr.shape == (4, 4), "transformation must be a 4x4 matrix."
        assert arr[3, 3] == 1.0, "Last element of transformation matrix must be 1.0."
        assert np.allclose(arr[3, :], [0.0, 0.0, 0.0, 1.0]), "Last row of transformation matrix must be [0, 0, 0, 1]."
        return v

    @property
    def transformation(self) -> np.ndarray:
        """The transformation necessary to transform coordinates from the object space to the tomogram space.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        return np.array(self.transformation_)

    @transformation.setter
    def transformation(self, value: np.ndarray) -> None:
        """Set the transformation matrix."""
        assert value.shape == (4, 4), "Transformation must be a 4x4 matrix."
        assert value[3, 3] == 1.0, "Last element of transformation matrix must be 1.0."
        assert np.allclose(value[3, :], [0.0, 0.0, 0.0, 1.0]), "Last row of transformation matrix must be [0, 0, 0, 1]."
        self.transformation_ = value.tolist()


class CopickObject:
    """Object that can be picked or segmented in a tomogram.

    Attributes:
        meta (PickableObject): Metadata for this object.
        root (CopickRoot): Reference to the root this object belongs to.
        name: Name of the object.
        is_particle: Whether this object should be represented by points (True) or segmentation masks (False).
        label: Numeric label/id for the object, as used in multilabel segmentation masks. Must be unique.
        color: RGBA color for the object.
        emdb_id: EMDB ID for the object.
        pdb_id: PDB ID for the object.
        map_threshold: Threshold to apply to the map when rendering the isosurface.
        radius: Radius of the particle, when displaying as a sphere.
    """

    def __init__(self, root: TCopickRoot, meta: PickableObject):
        """
        Args:
            root(CopickRoot): The copick project root.
            meta: The metadata for this object.
        """

        self.meta = meta
        self.root = root

    def __repr__(self):
        label = self.label if self.label is not None else "None"
        color = self.color if self.color is not None else "None"
        emdb_id = self.emdb_id if self.emdb_id is not None else "None"
        pdb_id = self.pdb_id if self.pdb_id is not None else "None"
        map_threshold = self.map_threshold if self.map_threshold is not None else "None"

        ret = (
            f"CopickObject(name={self.name}, is_particle={self.is_particle}, label={label}, color={color}, "
            f"emdb_id={emdb_id}, pdb_id={pdb_id}, threshold={map_threshold}) at {hex(id(self))}"
        )
        return ret

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def is_particle(self) -> bool:
        return self.meta.is_particle

    @property
    def label(self) -> Union[int, None]:
        return self.meta.label

    @property
    def color(self) -> Union[Tuple[int, int, int, int], None]:
        return self.meta.color

    @property
    def emdb_id(self) -> Union[str, None]:
        return self.meta.emdb_id

    @property
    def pdb_id(self) -> Union[str, None]:
        return self.meta.pdb_id

    @property
    def map_threshold(self) -> Union[float, None]:
        return self.meta.map_threshold

    @property
    def radius(self) -> Union[float, None]:
        return self.meta.radius

    def zarr(self) -> Union[None, MutableMapping]:
        """Override this method to return a zarr store for this object. Should return None if
        CopickObject.is_particle is False."""
        if not self.is_particle:
            return None

        raise NotImplementedError("zarr method must be implemented for particle objects.")


class CopickRoot:
    """Root of a copick project. Contains references to the runs and pickable objects.

    Attributes:
        config (CopickConfig): Configuration of the copick project.
        user_id: Unique identifier for the user.
        session_id: Unique identifier for the session.
        runs (List[CopickRun]): References to the runs for this project. Lazy loaded upon access.
        pickable_objects (List[CopickObject]): References to the pickable objects for this project.

    """

    def __init__(self, config: TCopickConfig):
        """
        Args:
            config (CopickConfig): Configuration of the copick project.
        """
        self.config = config
        self._runs: Optional[List[TCopickRun]] = None
        self._objects: Optional[List[TCopickObject]] = None

        # If runs are specified in the config, create them
        if config.runs is not None:
            self._runs = [CopickRun(self, CopickRunMeta(name=run_name)) for run_name in config.runs]

    def __repr__(self):
        lpo = None if self._objects is None else len(self._objects)
        lr = None if self._runs is None else len(self._runs)
        return f"CopickRoot(user_id={self.user_id}, len(pickable_objects)={lpo}, len(runs)={lr}) at {hex(id(self))}"

    @property
    def user_id(self) -> str:
        return self.config.user_id

    @user_id.setter
    def user_id(self, value: str) -> None:
        self.config.user_id = value

    @property
    def session_id(self) -> str:
        return self.config.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self.config.session_id = value

    def query(self) -> List[TCopickRun]:
        """Override this method to query for runs."""
        pass

    @property
    def runs(self) -> List[TCopickRun]:
        if self._runs is None:
            self._runs = self.query()

        return self._runs

    def get_run(self, name: str, **kwargs) -> Union[TCopickRun, None]:
        """Get run by name.

        Args:
            name: Name of the run to retrieve.
            **kwargs: Additional keyword arguments for the run metadata.

        Returns:
            CopickRun: The run with the given name, or None if not found.
        """
        # Random access
        if self._runs is None:
            clz, meta_clz = self._run_factory()
            rm = meta_clz(name=name, **kwargs)
            run = clz(self, meta=rm)

            if not run.ensure(create=False):
                return None
            else:
                return run

        # Access through index
        else:
            for run in self.runs:
                if run.name == name:
                    return run

        return None

    @property
    def pickable_objects(self) -> List[TCopickObject]:
        if self._objects is None:
            clz, meta_clz = self._object_factory()
            self._objects = [clz(self, meta=obj) for obj in self.config.pickable_objects]

        return self._objects

    def get_object(self, name: str) -> Union[TCopickObject, None]:
        """Get object by name.

        Args:
            name: Name of the object to retrieve.

        Returns:
            CopickObject: The object with the given name, or None if not found.
        """
        for obj in self.pickable_objects:
            if obj.name == name:
                return obj

        return None

    def refresh(self) -> None:
        """Refresh the list of runs."""
        self._runs = self.query()

    def new_run(self, name: str, **kwargs) -> TCopickRun:
        """Create a new run.

        Args:
            name: Name of the run to create.
            **kwargs: Additional keyword arguments for the run metadata.

        Returns:
            CopickRun: The newly created run.

        Raises:
            ValueError: If a run with the given name already exists.
        """
        if name in [r.name for r in self.runs]:
            raise ValueError(f"Run name {name} already exists.")

        clz, meta_clz = self._run_factory()
        rm = meta_clz(name=name, **kwargs)
        run = clz(self, meta=rm)

        # Append the run
        if self._runs is None:
            self._runs = []
        self._runs.append(run)

        run.ensure(create=True)

        return run

    def _run_factory(self) -> Tuple[Type[TCopickRun], Type["CopickRunMeta"]]:
        """Override this method to return the run class and run metadata class."""
        return CopickRun, CopickRunMeta

    def _object_factory(self) -> Tuple[Type[TCopickObject], Type["PickableObject"]]:
        """Override this method to return the object class and object metadata class."""
        return CopickObject, PickableObject


class CopickRunMeta(BaseModel):
    """Data model for run level metadata.

    Attributes:
        name: Name of the run.
    """

    name: str


class CopickRun:
    """Encapsulates all data pertaining to a physical location on a sample (i.e. typically one tilt series and the
    associated tomograms). This includes voxel spacings (of the reconstructed tomograms), picks, meshes, and
    segmentations.

    Attributes:
        meta (CopickRunMeta): Metadata for this run.
        root (CopickRoot): Reference to the root project this run belongs to.
        voxel_spacings (List[CopickVoxelSpacing]): Voxel spacings for this run. Either populated from config or lazily
            loaded when CopickRun.voxel_spacings is accessed **for the first time**.
        picks (List[CopickPicks]): Picks for this run. Either populated from config or lazily loaded when
            CopickRun.picks is accessed for **the first time**.
        meshes (List[CopickMesh]): Meshes for this run. Either populated from config or lazily loaded when
            CopickRun.meshes is accessed **for the first time**.
        segmentations (List[CopickSegmentation]): Segmentations for this run. Either populated from config or lazily
            loaded when CopickRun.segmentations is accessed **for the first time**.


    """

    def __init__(self, root: TCopickRoot, meta: CopickRunMeta, config: Optional[TCopickConfig] = None):
        self.meta = meta
        self.root = root
        self._voxel_spacings: Optional[List[TCopickVoxelSpacing]] = None
        """Voxel spacings for this run. Either populated from config or lazily loaded when CopickRun.voxel_spacings is
        accessed for the first time."""
        self._picks: Optional[List[TCopickPicks]] = None
        """Picks for this run. Either populated from config or lazily loaded when CopickRun.picks is
        accessed for the first time."""
        self._meshes: Optional[List[TCopickMesh]] = None
        """Meshes for this run. Either populated from config or lazily loaded when CopickRun.picks is
        accessed for the first time."""
        self._segmentations: Optional[List[TCopickSegmentation]] = None
        """Segmentations for this run. Either populated from config or lazily loaded when
        CopickRun.segmentations is accessed for the first time."""

        if config is not None:
            voxel_spacings_metas = [
                CopickVoxelSpacingMeta(run=self, voxel_size=vs, config=config) for vs in config.tomograms
            ]
            self._voxel_spacings = [CopickVoxelSpacing(run=self, meta=vs) for vs in voxel_spacings_metas]

            #####################
            # Picks from config #
            #####################
            # Select all available pre-picks for this run
            avail = config.available_pre_picks.keys()
            avail = [a for a in avail if a[0] == self.name]

            # Pre-defined picks
            for av in avail:
                object_name = av[1]
                prepicks = config.available_pre_picks[av]

                for pp in prepicks:
                    pm = CopickPicksFile(
                        pickable_object_name=object_name,
                        user_id=pp,
                        session_id="0",
                        run_name=self.name,
                    )
                    self._picks.append(CopickPicks(run=self, file=pm))

            ######################
            # Meshes from config #
            ######################
            for object_name, tool_names in config.available_pre_meshes.items():
                for mesh_tool in tool_names:
                    mm = CopickMeshMeta(pickable_object_name=object_name, user_id=mesh_tool, session_id="0")
                    com = CopickMesh(run=self, meta=mm)
                    self._meshes.append(com)

            #############################
            # Segmentations from config #
            #############################
            for seg_tool in config.available_pre_segmentations:
                sm = CopickSegmentationMeta(run=self, user_id=seg_tool, session_id="0")
                cos = CopickSegmentation(run=self, meta=sm)
                self._segmentations.append(cos)

    def __repr__(self):
        lvs = None if self._voxel_spacings is None else len(self._voxel_spacings)
        lpck = None if self._picks is None else len(self._picks)
        lmsh = None if self._meshes is None else len(self._meshes)
        lseg = None if self._segmentations is None else len(self._segmentations)
        ret = (
            f"CopickRun(name={self.name}, len(voxel_spacings)={lvs}, len(picks)={lpck}, len(meshes)={lmsh}, "
            f"len(segmentations)={lseg}) at {hex(id(self))}"
        )
        return ret

    @property
    def name(self):
        return self.meta.name

    @name.setter
    def name(self, value: str) -> None:
        self.meta.name = value

    def query_voxelspacings(self) -> List[TCopickVoxelSpacing]:
        """Override this method to query for voxel_spacings.

        Returns:
            List[CopickVoxelSpacing]: List of voxel spacings for this run.
        """
        raise NotImplementedError("query_voxelspacings must be implemented for CopickRun.")

    def query_picks(self) -> List[TCopickPicks]:
        """Override this method to query for picks.

        Returns:
            List[CopickPicks]: List of picks for this run.
        """
        raise NotImplementedError("query_picks must be implemented for CopickRun.")

    def query_meshes(self) -> List[TCopickMesh]:
        """Override this method to query for meshes.

        Returns:
            List[CopickMesh]: List of meshes for this run.
        """
        raise NotImplementedError("query_meshes must be implemented for CopickRun.")

    def query_segmentations(self) -> List[TCopickSegmentation]:
        """Override this method to query for segmentations.

        Returns:
            List[CopickSegmentation]: List of segmentations for this run.
        """
        raise NotImplementedError("query_segmentations must be implemented for CopickRun.")

    @property
    def voxel_spacings(self) -> List[TCopickVoxelSpacing]:
        if self._voxel_spacings is None:
            self._voxel_spacings = self.query_voxelspacings()

        return self._voxel_spacings

    def get_voxel_spacing(self, voxel_size: float, **kwargs) -> Union[TCopickVoxelSpacing, None]:
        """Get voxel spacing object by voxel size value.

        Args:
            voxel_size: Voxel size value to search for.
            **kwargs: Additional keyword arguments for the voxel spacing metadata.

        Returns:
            CopickVoxelSpacing: The voxel spacing object with the given voxel size value, or None if not found.
        """
        # Random access
        if self._voxel_spacings is None:
            clz, meta_clz = self._voxel_spacing_factory()
            vm = meta_clz(voxel_size=voxel_size, **kwargs)
            vs = clz(self, meta=vm)

            if not vs.ensure(create=False):
                return None
            else:
                return vs

        # Access through index
        else:
            for vs in self.voxel_spacings:
                if vs.voxel_size == voxel_size:
                    return vs

        return None

    @property
    def picks(self) -> List[TCopickPicks]:
        if self._picks is None:
            self._picks = self.query_picks()

        return self._picks

    def user_picks(self) -> List[TCopickPicks]:
        """Get all user generated picks (i.e. picks that have `CopickPicks.session_id != 0`).

        Returns:
            List[CopickPicks]: List of user-generated picks.
        """
        if self.root.config.user_id is None:
            return [p for p in self.picks if p.session_id != "0"]
        else:
            return self.get_picks(user_id=self.root.config.user_id)

    def tool_picks(self) -> List[TCopickPicks]:
        """Get all tool generated picks (i.e. picks that have `CopickPicks.session_id == 0`).

        Returns:
            List[CopickPicks]: List of tool-generated picks.
        """
        return [p for p in self.picks if p.session_id == "0"]

    def get_picks(self, object_name: str = None, user_id: str = None, session_id: str = None) -> List[TCopickPicks]:
        """Get picks by name, user_id or session_id (or combinations).

        Args:
            object_name: Name of the object to search for.
            user_id: User ID to search for.
            session_id: Session ID to search for.

        Returns:
            List[CopickPicks]: List of picks that match the search criteria.
        """
        ret = self.picks

        if object_name is not None:
            ret = [p for p in ret if p.pickable_object_name == object_name]

        if user_id is not None:
            ret = [p for p in ret if p.user_id == user_id]

        if session_id is not None:
            ret = [p for p in ret if p.session_id == session_id]

        return ret

    @property
    def meshes(self) -> List[TCopickMesh]:
        if self._meshes is None:
            self._meshes = self.query_meshes()

        return self._meshes

    def user_meshes(self) -> List[TCopickMesh]:
        """Get all user generated meshes (i.e. meshes that have `CopickMesh.session_id != 0`).

        Returns:
            List[CopickMesh]: List of user-generated meshes.
        """
        if self.root.config.user_id is None:
            return [m for m in self.meshes if m.session_id != "0"]
        else:
            return self.get_meshes(user_id=self.root.config.user_id)

    def tool_meshes(self) -> List[TCopickMesh]:
        """Get all tool generated meshes (i.e. meshes that have `CopickMesh.session_id == 0`).

        Returns:
            List[CopickMesh]: List of tool-generated meshes.
        """
        return [m for m in self.meshes if m.session_id == "0"]

    def get_meshes(self, object_name: str = None, user_id: str = None, session_id: str = None) -> List[TCopickMesh]:
        """Get meshes by name, user_id or session_id (or combinations).

        Args:
            object_name: Name of the object to search for.
            user_id: User ID to search for.
            session_id: Session ID to search for.

        Returns:
            List[CopickMesh]: List of meshes that match the search criteria.
        """
        ret = self.meshes

        if object_name is not None:
            ret = [m for m in ret if m.pickable_object_name == object_name]

        if user_id is not None:
            ret = [m for m in ret if m.user_id == user_id]

        if session_id is not None:
            ret = [m for m in ret if m.session_id == session_id]

        return ret

    @property
    def segmentations(self) -> List[TCopickSegmentation]:
        if self._segmentations is None:
            self._segmentations = self.query_segmentations()

        return self._segmentations

    def user_segmentations(self) -> List[TCopickSegmentation]:
        """Get all user generated segmentations (i.e. segmentations that have `CopickSegmentation.session_id != 0`).

        Returns:
            List[CopickSegmentation]: List of user-generated segmentations.
        """
        if self.root.config.user_id is None:
            return [s for s in self.segmentations if s.session_id != "0"]
        else:
            return self.get_segmentations(user_id=self.root.config.user_id)

    def tool_segmentations(self) -> List[TCopickSegmentation]:
        """Get all tool generated segmentations (i.e. segmentations that have `CopickSegmentation.session_id == 0`).

        Returns:
            List[CopickSegmentation]: List of tool-generated segmentations.
        """
        return [s for s in self.segmentations if s.session_id == "0"]

    def get_segmentations(
        self,
        user_id: str = None,
        session_id: str = None,
        is_multilabel: bool = None,
        name: str = None,
        voxel_size: float = None,
    ) -> List[TCopickSegmentation]:
        """Get segmentations by user_id, session_id, name, type or voxel_size (or combinations).

        Args:
            user_id: User ID to search for.
            session_id: Session ID to search for.
            is_multilabel: Whether the segmentation is multilabel or not.
            name: Name of the segmentation to search for.
            voxel_size: Voxel size to search for.

        Returns:
            List[CopickSegmentation]: List of segmentations that match the search criteria.
        """
        ret = self.segmentations

        if user_id is not None:
            ret = [s for s in ret if s.user_id == user_id]

        if session_id is not None:
            ret = [s for s in ret if s.session_id == session_id]

        if is_multilabel is not None:
            ret = [s for s in ret if s.is_multilabel == is_multilabel]

        if name is not None:
            ret = [s for s in ret if s.name == name]

        if voxel_size is not None:
            ret = [s for s in ret if s.voxel_size == voxel_size]

        return ret

    def new_voxel_spacing(self, voxel_size: float, **kwargs) -> TCopickVoxelSpacing:
        """Create a new voxel spacing object.

        Args:
            voxel_size: Voxel size value for the contained tomograms.
            **kwargs: Additional keyword arguments for the voxel spacing metadata.

        Returns:
            CopickVoxelSpacing: The newly created voxel spacing object.

        Raises:
            ValueError: If a voxel spacing with the given voxel size already exists for this run.
        """
        if voxel_size in [vs.voxel_size for vs in self.voxel_spacings]:
            raise ValueError(f"VoxelSpacing {voxel_size} already exists for this run.")

        clz, meta_clz = self._voxel_spacing_factory()

        vm = meta_clz(voxel_size=voxel_size, **kwargs)
        vs = clz(run=self, meta=vm)

        # Append the voxel spacing
        if self._voxel_spacings is None:
            self._voxel_spacings = []
        self._voxel_spacings.append(vs)

        # Ensure the voxel spacing record exists
        vs.ensure(create=True)

        return vs

    def _voxel_spacing_factory(self) -> Tuple[Type[TCopickVoxelSpacing], Type["CopickVoxelSpacingMeta"]]:
        """Override this method to return the voxel spacing class and voxel spacing metadata class."""
        return CopickVoxelSpacing, CopickVoxelSpacingMeta

    def new_picks(self, object_name: str, session_id: str, user_id: Optional[str] = None) -> TCopickPicks:
        """Create a new picks object.

        Args:
            object_name: Name of the object to pick.
            session_id: Session ID for the picks.
            user_id: User ID for the picks.

        Returns:
            CopickPicks: The newly created picks object.

        Raises:
            ValueError: If picks for the given object name, session ID and user ID already exist, if the object name
                is not found in the pickable objects, or if the user ID is not set in the root config or supplied.
        """
        if object_name not in [o.name for o in self.root.config.pickable_objects]:
            raise ValueError(f"Object name {object_name} not found in pickable objects.")

        uid = self.root.config.user_id

        if user_id is not None:
            uid = user_id

        if uid is None:
            raise ValueError("User ID must be set in the root config or supplied to new_picks.")

        if self.get_picks(object_name=object_name, session_id=session_id, user_id=uid):
            raise ValueError(f"Picks for {object_name} by user/tool {uid} already exist in session {session_id}.")

        pm = CopickPicksFile(
            pickable_object_name=object_name,
            user_id=uid,
            session_id=session_id,
            run_name=self.name,
        )

        clz = self._picks_factory()

        picks = clz(run=self, file=pm)

        if self._picks is None:
            self._picks = []
        self._picks.append(picks)

        # Create the picks file
        picks.store()

        return picks

    def _picks_factory(self) -> Type[TCopickPicks]:
        """Override this method to return the picks class."""
        return CopickPicks

    def new_mesh(self, object_name: str, session_id: str, user_id: Optional[str] = None, **kwargs) -> TCopickMesh:
        """Create a new mesh object.

        Args:
            object_name: Name of the object to mesh.
            session_id: Session ID for the mesh.
            user_id: User ID for the mesh.
            **kwargs: Additional keyword arguments for the mesh metadata.

        Returns:
            CopickMesh: The newly created mesh object.

        Raises:
            ValueError: If a mesh for the given object name, session ID and user ID already exist, if the object name
                is not found in the pickable objects, or if the user ID is not set in the root config or supplied.
        """
        if object_name not in [o.name for o in self.root.config.pickable_objects]:
            raise ValueError(f"Object name {object_name} not found in pickable objects.")

        uid = self.root.config.user_id

        if user_id is not None:
            uid = user_id

        if uid is None:
            raise ValueError("User ID must be set in the root config or supplied to new_mesh.")

        if self.get_meshes(object_name=object_name, session_id=session_id, user_id=uid):
            raise ValueError(f"Mesh for {object_name} by user/tool {uid} already exist in session {session_id}.")

        clz, meta_clz = self._mesh_factory()

        mm = meta_clz(
            pickable_object_name=object_name,
            user_id=uid,
            session_id=session_id,
            **kwargs,
        )

        # Need to create an empty trimesh.Trimesh object first, because empty scenes can't be exported.
        tmesh = trimesh.Trimesh()
        scene = tmesh.scene()

        mesh = clz(run=self, meta=mm, mesh=scene)

        if self._meshes is None:
            self._meshes = []
        self._meshes.append(mesh)

        # Create the mesh file
        mesh.store()

        return mesh

    def _mesh_factory(self) -> Tuple[Type[TCopickMesh], Type["CopickMeshMeta"]]:
        """Override this method to return the mesh class and mesh metadata."""
        return CopickMesh, CopickMeshMeta

    def new_segmentation(
        self,
        voxel_size: float,
        name: str,
        session_id: str,
        is_multilabel: bool,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> TCopickSegmentation:
        """Create a new segmentation object.

        Args:
            voxel_size: Voxel size for the segmentation.
            name: Name of the segmentation.
            session_id: Session ID for the segmentation.
            is_multilabel: Whether the segmentation is multilabel or not.
            user_id: User ID for the segmentation.
            **kwargs: Additional keyword arguments for the segmentation metadata.

        Returns:
            CopickSegmentation: The newly created segmentation object.

        Raises:
            ValueError: If a segmentation for the given name, session ID, user ID, voxel size and multilabel flag already
                exist, if the object name is not found in the pickable objects, if the voxel size is not found in the
                voxel spacings, or if the user ID is not set in the root config or supplied.
        """
        if not is_multilabel and name not in [o.name for o in self.root.config.pickable_objects]:
            raise ValueError(f"Object name {name} not found in pickable objects.")

        if voxel_size not in [vs.voxel_size for vs in self.voxel_spacings]:
            raise ValueError(f"VoxelSpacing {voxel_size} not found in voxel spacings for run {self.name}.")

        uid = self.root.config.user_id

        if user_id is not None:
            uid = user_id

        if uid is None:
            raise ValueError("User ID must be set in the root config or supplied to new_segmentation.")

        if self.get_segmentations(
            session_id=session_id,
            user_id=uid,
            name=name,
            is_multilabel=is_multilabel,
            voxel_size=voxel_size,
        ):
            raise ValueError(
                f"Segmentation by user/tool {uid} already exist in session {session_id} with name {name}, voxel size of {voxel_size}, and has a multilabel flag of {is_multilabel}.",
            )

        clz, meta_clz = self._segmentation_factory()

        sm = meta_clz(
            is_multilabel=is_multilabel,
            voxel_size=voxel_size,
            user_id=uid,
            session_id=session_id,
            name=name,
            **kwargs,
        )
        seg = clz(run=self, meta=sm)

        if self._segmentations is None:
            self._segmentations = []

        self._segmentations.append(seg)

        # Create the zarr store for this segmentation
        _ = seg.zarr()

        return seg

    def _segmentation_factory(self) -> Tuple[Type[TCopickSegmentation], Type["CopickSegmentationMeta"]]:
        """Override this method to return the segmentation class and segmentation metadata class."""
        return CopickSegmentation, CopickSegmentationMeta

    def refresh_voxel_spacings(self) -> None:
        """Refresh the voxel spacings."""
        self._voxel_spacings = self.query_voxelspacings()

    def refresh_picks(self) -> None:
        """Refresh the picks."""
        self._picks = self.query_picks()

    def refresh_meshes(self) -> None:
        """Refresh the meshes."""
        self._meshes = self.query_meshes()

    def refresh_segmentations(self) -> None:
        """Refresh the segmentations."""
        self._segmentations = self.query_segmentations()

    def refresh(self) -> None:
        """Refresh all child types."""
        self.refresh_voxel_spacings()
        self.refresh_picks()
        self.refresh_meshes()
        self.refresh_segmentations()

    def ensure(self, create: bool = False) -> bool:
        """Check if the run record exists, optionally create it if it does not.

        Args:
            create: Whether to create the run record if it does not exist.

        Returns:
            bool: True if the run record exists, False otherwise.
        """
        raise NotImplementedError("ensure must be implemented for CopickRun.")


class CopickVoxelSpacingMeta(BaseModel):
    """Data model for voxel spacing metadata.

    Attributes:
        voxel_size: Voxel size in angstrom, rounded to the third decimal.
    """

    voxel_size: float


class CopickVoxelSpacing:
    """Encapsulates all data pertaining to a specific voxel spacing. This includes the tomograms and feature maps at
    this voxel spacing.

    Attributes:
        run (CopickRun): Reference to the run this voxel spacing belongs to.
        meta (CopickVoxelSpacingMeta): Metadata for this voxel spacing.
        tomograms (List[CopickTomogram]): Tomograms for this voxel spacing. Either populated from config or lazily loaded
            when CopickVoxelSpacing.tomograms is accessed **for the first time**.
    """

    def __init__(self, run: TCopickRun, meta: CopickVoxelSpacingMeta, config: Optional[TCopickConfig] = None):
        """
        Args:
            run: Reference to the run this voxel spacing belongs to.
            meta: Metadata for this voxel spacing.
            config: Configuration of the copick project.
        """
        self.run = run
        self.meta = meta

        self._tomograms: Optional[List[TCopickTomogram]] = None
        """References to the tomograms for this voxel spacing."""

        if config is not None:
            tomo_metas = [CopickTomogramMeta(tomo_type=tt) for tt in config.tomograms[self.voxel_size]]
            self._tomograms = [CopickTomogram(voxel_spacing=self, meta=tm, config=config) for tm in tomo_metas]

    def __repr__(self):
        lts = None if self._tomograms is None else len(self._tomograms)
        return f"CopickVoxelSpacing(voxel_size={self.voxel_size}, len(tomograms)={lts}) at {hex(id(self))}"

    @property
    def voxel_size(self) -> float:
        return self.meta.voxel_size

    def query_tomograms(self) -> List[TCopickTomogram]:
        """Override this method to query for tomograms."""
        raise NotImplementedError("query_tomograms must be implemented for CopickVoxelSpacing.")

    @property
    def tomograms(self) -> List[TCopickTomogram]:
        if self._tomograms is None:
            self._tomograms = self.query_tomograms()

        return self._tomograms

    def get_tomogram(self, tomo_type: str, **kwargs) -> Union[TCopickTomogram, None]:
        """Get tomogram by type.

        Args:
            tomo_type: Type of the tomogram to retrieve.

        Returns:
            CopickTomogram: The tomogram with the given type, or `None` if not found.
        """
        for tomo in self.tomograms:
            if tomo.tomo_type == tomo_type:
                return tomo
        return None

    def refresh_tomograms(self) -> None:
        """Refresh `CopickVoxelSpacing.tomograms` from storage."""
        self._tomograms = self.query_tomograms()

    def refresh(self) -> None:
        """Refresh `CopickVoxelSpacing.tomograms` from storage."""
        self.refresh_tomograms()

    def new_tomogram(self, tomo_type: str, **kwargs) -> TCopickTomogram:
        """Create a new tomogram object, also creates the Zarr-store in the storage backend.

        Args:
            tomo_type: Type of the tomogram to create.
            **kwargs: Additional keyword arguments for the tomogram metadata.

        Returns:
            CopickTomogram: The newly created tomogram object.

        Raises:
            ValueError: If a tomogram with the given type already exists for this voxel spacing.
        """
        if tomo_type in [tomo.tomo_type for tomo in self.tomograms]:
            raise ValueError(f"Tomogram type {tomo_type} already exists for this voxel spacing.")

        clz, meta_clz = self._tomogram_factory()

        tm = meta_clz(tomo_type=tomo_type, **kwargs)
        tomo = clz(voxel_spacing=self, meta=tm)

        # Append the tomogram
        if self._tomograms is None:
            self._tomograms = []
        self._tomograms.append(tomo)

        # Create the zarr store for this tomogram
        _ = tomo.zarr()

        return tomo

    def _tomogram_factory(self) -> Tuple[Type[TCopickTomogram], Type["CopickTomogramMeta"]]:
        """Override this method to return the tomogram class."""
        return CopickTomogram, CopickTomogramMeta

    def ensure(self, create: bool = False) -> bool:
        """Override to check if the voxel spacing record exists, optionally create it if it does not.

        Args:
            create: Whether to create the voxel spacing record if it does not exist.

        Returns:
            bool: True if the voxel spacing record exists, False otherwise.
        """
        raise NotImplementedError("ensure must be implemented for CopickVoxelSpacing.")


class CopickTomogramMeta(BaseModel):
    """Data model for tomogram metadata.

    Attributes:
        tomo_type: Type of the tomogram.
    """

    tomo_type: str


class CopickTomogram:
    """Encapsulates all data pertaining to a specific tomogram. This includes the features for this tomogram and the
    associated Zarr-store.

    Attributes:
        voxel_spacing (CopickVoxelSpacing): Reference to the voxel spacing this tomogram belongs to.
        meta (CopickTomogramMeta): Metadata for this tomogram.
        features (List[CopickFeatures]): Features for this tomogram. Either populated from config or lazily loaded when
            `CopickTomogram.features` is accessed **for the first time**.
        tomo_type (str): Type of the tomogram.
    """

    def __init__(
        self,
        voxel_spacing: TCopickVoxelSpacing,
        meta: CopickTomogramMeta,
        config: Optional[TCopickConfig] = None,
    ):
        self.meta = meta
        self.voxel_spacing = voxel_spacing

        self._features: Optional[List[TCopickFeatures]] = None
        """Features for this tomogram."""

        if config is not None and self.tomo_type in config.features[self.voxel_spacing.voxel_size]:
            feat_metas = [CopickFeaturesMeta(tomo_type=self.tomo_type, feature_type=ft) for ft in config.feature_types]
            self._features = [CopickFeatures(tomogram=self, meta=fm) for fm in feat_metas]

    def __repr__(self):
        lft = None if self._features is None else len(self._features)
        return f"CopickTomogram(tomo_type={self.tomo_type}, len(features)={lft}) at {hex(id(self))}"

    @property
    def tomo_type(self) -> str:
        return self.meta.tomo_type

    @property
    def features(self) -> List[TCopickFeatures]:
        if self._features is None:
            self._features = self.query_features()

        return self._features

    @features.setter
    def features(self, value: List[TCopickFeatures]) -> None:
        """Set the features."""
        self._features = value

    def get_features(self, feature_type: str) -> Union[TCopickFeatures, None]:
        """Get feature maps by type.

        Args:
            feature_type: Type of the feature map to retrieve.

        Returns:
            CopickFeatures: The feature map with the given type, or `None` if not found.
        """
        for feat in self.features:
            if feat.feature_type == feature_type:
                return feat
        return None

    def new_features(self, feature_type: str, **kwargs) -> TCopickFeatures:
        """Create a new feature map object. Also creates the Zarr-store for the map in the storage backend.

        Args:
            feature_type: Type of the feature map to create.
            **kwargs: Additional keyword arguments for the feature map metadata.

        Returns:
            CopickFeatures: The newly created feature map object.

        Raises:
            ValueError: If a feature map with the given type already exists for this tomogram.
        """
        if feature_type in [f.feature_type for f in self.features]:
            raise ValueError(f"Feature type {feature_type} already exists for this tomogram.")

        clz, meta_clz = self._feature_factory()

        fm = meta_clz(tomo_type=self.tomo_type, feature_type=feature_type, **kwargs)
        feat = clz(tomogram=self, meta=fm)

        # Append the feature set
        if self._features is None:
            self._features = []

        self._features.append(feat)

        # Create the zarr store for this feature set
        _ = feat.zarr()

        return feat

    def _feature_factory(self) -> Tuple[Type[TCopickFeatures], Type["CopickFeaturesMeta"]]:
        """Override this method to return the features class and features metadata class."""
        return CopickFeatures, CopickFeaturesMeta

    def query_features(self) -> List[TCopickFeatures]:
        """Override this method to query for features."""
        raise NotImplementedError("query_features must be implemented for CopickTomogram.")

    def refresh_features(self) -> None:
        """Refresh `CopickTomogram.features` from storage."""
        self._features = self.query_features()

    def refresh(self) -> None:
        """Refresh `CopickTomogram.features` from storage."""
        self.refresh_features()

    def zarr(self) -> MutableMapping:
        """Override to return the Zarr store for this tomogram. Also needs to handle creating the store if it
        doesn't exist."""
        raise NotImplementedError("zarr must be implemented for CopickTomogram.")


class CopickFeaturesMeta(BaseModel):
    """Data model for feature map metadata.

    Attributes:
        tomo_type: Type of the tomogram that the features were computed on.
        feature_type: Type of the features contained.
    """

    tomo_type: str
    feature_type: str


class CopickFeatures:
    """Encapsulates all data pertaining to a specific feature map, i.e. the Zarr-store for the feature map.

    Attributes:
        tomogram (CopickTomogram): Reference to the tomogram this feature map belongs to.
        meta (CopickFeaturesMeta): Metadata for this feature map.
        tomo_type (str): Type of the tomogram that the features were computed on.
        feature_type (str): Type of the features contained.
    """

    def __init__(self, tomogram: TCopickTomogram, meta: CopickFeaturesMeta):
        """

        Args:
            tomogram: Reference to the tomogram this feature map belongs to.
            meta: Metadata for this feature map.
        """
        self.meta: CopickFeaturesMeta = meta
        self.tomogram: TCopickTomogram = tomogram

    def __repr__(self):
        return f"CopickFeatures(tomo_type={self.tomo_type}, feature_type={self.feature_type}) at {hex(id(self))}"

    @property
    def tomo_type(self) -> str:
        return self.meta.tomo_type

    @property
    def feature_type(self) -> str:
        return self.meta.feature_type

    def zarr(self) -> MutableMapping:
        """Override to return the Zarr store for this feature set. Also needs to handle creating the store if it
        doesn't exist."""
        raise NotImplementedError("zarr must be implemented for CopickFeatures.")


class CopickPicksFile(BaseModel):
    """Datamodel for a collection of locations, orientations and other metadata for one pickable object.

    Attributes:
        pickable_object_name: Pickable object name from CopickConfig.pickable_objects[X].name
        user_id: Unique identifier for the user or tool name.
        session_id: Unique identifier for the pick session (prevent race if they run multiple instances of napari,
            ChimeraX, etc.) If it is 0, this pick was generated by a tool.
        run_name: Name of the run this pick belongs to.
        voxel_spacing: Voxel spacing for the tomogram this pick belongs to.
        unit: Unit for the location of the pick.
        points (List[CopickPoint]): References to the points for this pick.
        trust_orientation: Flag to indicate if the angles are known for this pick or should be ignored.

    """

    pickable_object_name: str
    user_id: str
    session_id: Union[str, Literal["0"]]
    run_name: Optional[str]
    voxel_spacing: Optional[float]
    unit: str = "angstrom"
    points: Optional[List[TCopickPoint]] = None
    trust_orientation: Optional[bool] = True


class CopickPicks:
    """Encapsulates all data pertaining to a specific set of picked points. This includes the locations, orientations,
    and other metadata for the set of points.

    Attributes:
        run (CopickRun): Reference to the run this pick belongs to.
        meta (CopickPicksFile): Metadata for this pick.
        points (List[CopickPoint]): Points for this pick. Either populated from storage or lazily loaded when
            `CopickPicks.points` is accessed **for the first time**.
        from_tool (bool): Flag to indicate if this pick was generated by a tool.
        pickable_object_name (str): Pickable object name from `CopickConfig.pickable_objects[...].name`
        user_id (str): Unique identifier for the user or tool name.
        session_id (str): Unique identifier for the pick session
        trust_orientation (bool): Flag to indicate if the angles are known for this pick or should be ignored.
        color: Color of the pickable object this pick belongs to.
    """

    def __init__(self, run: TCopickRun, file: CopickPicksFile):
        """
        Args:
            run: Reference to the run this pick belongs to.
            file: Metadata for this set of points.
        """
        self.meta: CopickPicksFile = file
        self.run: TCopickRun = run

    def __repr__(self):
        lpt = None if self.meta.points is None else len(self.meta.points)
        ret = (
            f"CopickPicks(pickable_object_name={self.pickable_object_name}, user_id={self.user_id}, "
            f"session_id={self.session_id}, len(points)={lpt}) at {hex(id(self))}"
        )
        return ret

    def _load(self) -> CopickPicksFile:
        """Override this method to load points from a RESTful interface or filesystem."""
        raise NotImplementedError("load must be implemented for CopickPicks.")

    def _store(self):
        """Override this method to store points with a RESTful interface or filesystem. Also needs to handle creating
        the file if it doesn't exist."""
        raise NotImplementedError("store must be implemented for CopickPicks.")

    def load(self) -> CopickPicksFile:
        """Load the points from storage.

        Returns:
            CopickPicksFile: The loaded points.
        """
        self.meta = self._load()

        return self.meta

    def store(self):
        """Store the points (set using `CopickPicks.points` property)."""
        self._store()

    @property
    def from_tool(self) -> bool:
        return self.session_id == "0"

    @property
    def pickable_object_name(self) -> str:
        return self.meta.pickable_object_name

    @property
    def user_id(self) -> str:
        return self.meta.user_id

    @property
    def session_id(self) -> Union[str, Literal["0"]]:
        return self.meta.session_id

    @property
    def points(self) -> List[TCopickPoint]:
        if self.meta.points is None:
            self.meta = self.load()

        return self.meta.points

    @points.setter
    def points(self, value: List[TCopickPoint]) -> None:
        self.meta.points = value

    @property
    def trust_orientation(self) -> bool:
        return self.meta.trust_orientation

    @property
    def color(self) -> Union[Tuple[int, int, int, int], None]:
        if self.run.root.get_object(self.pickable_object_name) is None:
            raise ValueError(f"{self.pickable_object_name} is not a recognized object name (run: {self.run.name}).")

        return self.run.root.get_object(self.pickable_object_name).color

    def refresh(self) -> None:
        """Refresh the points from storage."""
        self.meta = self.load()


class CopickMeshMeta(BaseModel):
    """Data model for mesh metadata.

    Attributes:
        pickable_object_name: Pickable object name from `CopickConfig.pickable_objects[...].name`
        user_id: Unique identifier for the user or tool name.
        session_id: Unique identifier for the pick session. If it is 0, this pick was generated by a tool.
    """

    pickable_object_name: str
    user_id: str
    session_id: Union[str, Literal["0"]]


class CopickMesh:
    """Encapsulates all data pertaining to a specific mesh. This includes the mesh (`trimesh.parent.Geometry`) and other
    metadata.

    Attributes:
        run (CopickRun): Reference to the run this mesh belongs to.
        meta (CopickMeshMeta): Metadata for this mesh.
        mesh (trimesh.parent.Geometry): Mesh for this pick. Either populated from storage or lazily loaded when
            `CopickMesh.mesh` is accessed **for the first time**.
        from_tool (bool): Flag to indicate if this pick was generated by a tool.
        from_user (bool): Flag to indicate if this pick was generated by a user.
        pickable_object_name (str): Pickable object name from `CopickConfig.pickable_objects[...].name`
        user_id (str): Unique identifier for the user or tool name.
        session_id (str): Unique identifier for the pick session
        color: Color of the pickable object this pick belongs to.
    """

    def __init__(self, run: TCopickRun, meta: CopickMeshMeta, mesh: Optional[Geometry] = None):
        self.meta: CopickMeshMeta = meta
        self.run: TCopickRun = run

        if mesh is not None:
            self._mesh = mesh
        else:
            self._mesh = None

    def __repr__(self):
        ret = (
            f"CopickMesh(pickable_object_name={self.pickable_object_name}, user_id={self.user_id}, "
            f"session_id={self.session_id}) at {hex(id(self))}"
        )
        return ret

    @property
    def pickable_object_name(self) -> str:
        return self.meta.pickable_object_name

    @property
    def user_id(self) -> str:
        return self.meta.user_id

    @property
    def session_id(self) -> Union[str, Literal["0"]]:
        return self.meta.session_id

    @property
    def color(self):
        return self.run.root.get_object(self.pickable_object_name).color

    def _load(self) -> Geometry:
        """Override this method to load mesh from a RESTful interface or filesystem."""
        raise NotImplementedError("load must be implemented for CopickMesh.")

    def _store(self):
        """Override this method to store mesh with a RESTful interface or filesystem. Also needs to handle creating
        the file if it doesn't exist."""
        raise NotImplementedError("store must be implemented for CopickMesh.")

    def load(self) -> Geometry:
        """Load the mesh from storage.

        Returns:
            trimesh.parent.Geometry: The loaded mesh.
        """
        self._mesh = self._load()

        return self._mesh

    def store(self):
        """Store the mesh."""
        self._store()

    @property
    def mesh(self) -> Geometry:
        if self._mesh is None:
            self._mesh = self.load()

        return self._mesh

    @mesh.setter
    def mesh(self, value: Geometry) -> None:
        self._mesh = value

    @property
    def from_user(self) -> bool:
        return self.session_id != "0"

    @property
    def from_tool(self) -> bool:
        return self.session_id == "0"

    def refresh(self) -> None:
        """Refresh `CopickMesh.mesh` from storage."""
        self._mesh = self.load()


class CopickSegmentationMeta(BaseModel):
    """Datamodel for segmentation metadata.

    Attributes:
        user_id: Unique identifier for the user or tool name.
        session_id: Unique identifier for the segmentation session. If it is 0, this segmentation was generated by a
            tool.
        name: Pickable Object name or multilabel name of the segmentation.
        is_multilabel: Flag to indicate if this is a multilabel segmentation. If False, it is a single label
            segmentation.
        voxel_size: Voxel size in angstrom of the tomogram this segmentation belongs to. Rounded to the third decimal.
    """

    user_id: str
    session_id: Union[str, Literal["0"]]
    name: str
    is_multilabel: bool
    voxel_size: float


class CopickSegmentation:
    """Encapsulates all data pertaining to a specific segmentation. This includes the Zarr-store for the segmentation
    and other metadata.

    Attributes:
        run (CopickRun): Reference to the run this segmentation belongs to.
        meta (CopickSegmentationMeta): Metadata for this segmentation.
        zarr (MutableMapping): Zarr store for this segmentation. Either populated from storage or lazily loaded when
            `CopickSegmentation.zarr` is accessed **for the first time**.
        from_tool (bool): Flag to indicate if this segmentation was generated by a tool.
        from_user (bool): Flag to indicate if this segmentation was generated by a user.
        user_id (str): Unique identifier for the user or tool name.
        session_id (str): Unique identifier for the segmentation session
        is_multilabel (bool): Flag to indicate if this is a multilabel segmentation. If False, it is a single label
            segmentation.
        voxel_size (float): Voxel size of the tomogram this segmentation belongs to.
        name (str): Pickable Object name or multilabel name of the segmentation.
        color: Color of the pickable object this segmentation belongs to.
    """

    def __init__(self, run: TCopickRun, meta: CopickSegmentationMeta):
        """

        Args:
            run: Reference to the run this segmentation belongs to.
            meta: Metadata for this segmentation.
        """
        self.meta: CopickSegmentationMeta = meta
        self.run: TCopickRun = run

    def __repr__(self):
        ret = (
            f"CopickSegmentation(user_id={self.user_id}, session_id={self.session_id}, name={self.name}, "
            f"is_multilabel={self.is_multilabel}, voxel_size={self.voxel_size}) at {hex(id(self))}"
        )
        return ret

    @property
    def user_id(self) -> str:
        return self.meta.user_id

    @property
    def session_id(self) -> Union[str, Literal["0"]]:
        return self.meta.session_id

    @property
    def from_tool(self) -> bool:
        return self.session_id == "0"

    @property
    def from_user(self) -> bool:
        return self.session_id != "0"

    @property
    def is_multilabel(self) -> bool:
        return self.meta.is_multilabel

    @property
    def voxel_size(self) -> float:
        return self.meta.voxel_size

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def color(self):
        if self.is_multilabel:
            return [128, 128, 128, 0]
        else:
            return self.run.root.get_object(self.name).color

    def zarr(self) -> MutableMapping:
        """Override to return the Zarr store for this segmentation. Also needs to handle creating the store if it
        doesn't exist."""
        raise NotImplementedError("zarr must be implemented for CopickSegmentation.")
