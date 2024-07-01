import concurrent.futures
import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import fsspec
import trimesh
import zarr
from fsspec import AbstractFileSystem
from trimesh.parent import Geometry

from copick.impl.overlay import (
    CopickFeaturesOverlay,
    CopickMeshOverlay,
    CopickObjectOverlay,
    CopickPicksOverlay,
    CopickRunOverlay,
    CopickSegmentationOverlay,
    CopickTomogramOverlay,
    CopickVoxelSpacingOverlay,
)
from copick.models import (
    CopickConfig,
    CopickFeatures,
    CopickFeaturesMeta,
    CopickMeshMeta,
    CopickPicksFile,
    CopickRoot,
    CopickRunMeta,
    CopickSegmentationMeta,
    CopickTomogramMeta,
    CopickVoxelSpacingMeta,
    PickableObject,
)


class CopickConfigFSSpec(CopickConfig):
    """Copick configuration for fsspec-based storage.

    Attributes:
        overlay_root (str): The root URL for the overlay storage.
        static_root (Optional[str]): The root URL for the static storage.
        overlay_fs_args (Optional[Dict[str, Any]]): Additional arguments for the overlay filesystem.
        static_fs_args (Optional[Dict[str, Any]]): Additional arguments for the static filesystem.
    """

    config_type: str = "filesystem"
    overlay_root: str
    static_root: Optional[str]

    overlay_fs_args: Optional[Dict[str, Any]] = {}
    static_fs_args: Optional[Dict[str, Any]] = {}


class CopickPicksFSSpec(CopickPicksOverlay):
    """CopickPicks class backed by fsspec storage.

    Attributes:
        path (str): The path to the picks file.
        directory (str): The directory containing the picks file.
        fs (AbstractFileSystem): The filesystem containing the picks file.
    """

    run: "CopickRunFSSpec"

    @property
    def path(self) -> str:
        if self.read_only:
            return f"{self.run.static_path}/Picks/{self.user_id}_{self.session_id}_{self.pickable_object_name}.json"
        else:
            return f"{self.run.overlay_path}/Picks/{self.user_id}_{self.session_id}_{self.pickable_object_name}.json"

    @property
    def directory(self) -> str:
        if self.read_only:
            return f"{self.run.static_path}/Picks/"
        else:
            return f"{self.run.overlay_path}/Picks/"

    @property
    def fs(self) -> AbstractFileSystem:
        return self.run.fs_static if self.read_only else self.run.fs_overlay

    def _load(self) -> CopickPicksFile:
        if not self.fs.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")

        with self.fs.open(self.path, "r") as f:
            data = json.load(f)

        return CopickPicksFile(**data)

    def _store(self) -> None:
        if not self.fs.exists(self.directory):
            self.fs.makedirs(self.directory, exist_ok=True)

        with self.fs.open(self.path, "w") as f:
            json.dump(self.meta.dict(), f, indent=4)


class CopickMeshFSSpec(CopickMeshOverlay):
    """CopickMesh class backed by fspec storage.

    Attributes:
        path (str): The path to the mesh file.
        directory (str): The directory containing the mesh file.
        fs (AbstractFileSystem): The filesystem containing the mesh file.
    """

    run: "CopickRunFSSpec"

    @property
    def path(self) -> str:
        if self.read_only:
            return f"{self.run.static_path}/Meshes/{self.user_id}_{self.session_id}_{self.pickable_object_name}.glb"
        else:
            return f"{self.run.overlay_path}/Meshes/{self.user_id}_{self.session_id}_{self.pickable_object_name}.glb"

    @property
    def directory(self) -> str:
        if self.read_only:
            return f"{self.run.static_path}/Meshes/"
        else:
            return f"{self.run.overlay_path}/Meshes/"

    @property
    def fs(self) -> AbstractFileSystem:
        return self.run.fs_static if self.read_only else self.run.fs_overlay

    def _load(self) -> Geometry:
        if not self.fs.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")

        with self.fs.open(self.path, "rb") as f:
            scene = trimesh.load(f, file_type="glb")

        return scene

    def _store(self):
        if not self.fs.exists(self.directory):
            self.fs.makedirs(self.directory, exist_ok=True)

        with self.fs.open(self.path, "wb") as f:
            _ = self._mesh.export(f, file_type="glb")


class CopickSegmentationFSSpec(CopickSegmentationOverlay):
    """CopickSegmentation class backed by fsspec storage.

    Attributes:
        filename (str): The filename of the segmentation file.
        path (str): The path to the segmentation file.
        fs (AbstractFileSystem): The filesystem containing the segmentation file.
    """

    run: "CopickRunFSSpec"

    @property
    def filename(self) -> str:
        if self.is_multilabel:
            return f"{self.voxel_size:.3f}_{self.user_id}_{self.session_id}_{self.name}-multilabel.zarr"
        else:
            return f"{self.voxel_size:.3f}_{self.user_id}_{self.session_id}_{self.name}.zarr"

    @property
    def path(self) -> str:
        if self.read_only:
            return f"{self.run.static_path}/Segmentations/{self.filename}"
        else:
            return f"{self.run.overlay_path}/Segmentations/{self.filename}"

    @property
    def fs(self) -> AbstractFileSystem:
        return self.run.fs_static if self.read_only else self.run.fs_overlay

    def zarr(self) -> zarr.storage.FSStore:
        """Get the zarr store for the segmentation object.

        Returns:
            zarr.storage.FSStore: The zarr store for the segmentation object.
        """
        if self.read_only:
            mode = "r"
            create = False
        else:
            mode = "w"
            create = not self.fs.exists(self.path)

        return zarr.storage.FSStore(
            self.path,
            fs=self.fs,
            mode=mode,
            key_separator="/",
            dimension_separator="/",
            create=create,
        )


class CopickFeaturesFSSpec(CopickFeaturesOverlay):
    """CopickFeatures class backed by fsspec storage.

    Attributes:
        path (str): The path to the features file.
        fs (AbstractFileSystem): The filesystem containing the features file.
    """

    tomogram: "CopickTomogramFSSpec"

    @property
    def path(self) -> str:
        if self.read_only:
            return f"{self.tomogram.static_stem}_{self.feature_type}_features.zarr"
        else:
            return f"{self.tomogram.overlay_stem}_{self.feature_type}_features.zarr"

    @property
    def fs(self) -> AbstractFileSystem:
        return self.tomogram.fs_static if self.read_only else self.tomogram.fs_overlay

    def zarr(self) -> zarr.storage.FSStore:
        """Get the zarr store for the features object.

        Returns:
            zarr.storage.FSStore: The zarr store for the features object.
        """
        if self.read_only:
            mode = "r"
            create = False
        else:
            mode = "w"
            create = not self.fs.exists(self.path)

        return zarr.storage.FSStore(
            self.path,
            fs=self.fs,
            mode=mode,
            key_separator="/",
            dimension_separator="/",
            create=create,
        )


class CopickTomogramFSSpec(CopickTomogramOverlay):
    """CopickTomogram class backed by fsspec storage.

    Attributes:
        static_path (str): The path to the tomogram on the static source.
        overlay_path (str): The path to the tomogram on the overlay source.
        static_stem (str): The stem of the tomogram on the static source.
        overlay_stem (str): The stem of the tomogram on the overlay source.
        fs_static (AbstractFileSystem): The filesystem containing the tomogram on the static source.
        fs_overlay (AbstractFileSystem): The filesystem containing the tomogram on the overlay source.
        static_is_overlay (bool): Whether the static and overlay sources are the same.
    """

    voxel_spacing: "CopickVoxelSpacingFSSpec"

    def _feature_factory(self) -> Tuple[Type[CopickFeatures], Type["CopickFeaturesMeta"]]:
        return CopickFeaturesFSSpec, CopickFeaturesMeta

    @property
    def static_path(self) -> str:
        return f"{self.voxel_spacing.static_path}/{self.tomo_type}.zarr"

    @property
    def overlay_path(self) -> str:
        return f"{self.voxel_spacing.overlay_path}/{self.tomo_type}.zarr"

    @property
    def static_stem(self) -> str:
        return f"{self.voxel_spacing.static_path}/{self.tomo_type}"

    @property
    def overlay_stem(self) -> str:
        return f"{self.voxel_spacing.overlay_path}/{self.tomo_type}"

    @property
    def fs_static(self) -> AbstractFileSystem:
        return self.voxel_spacing.fs_static

    @property
    def fs_overlay(self) -> AbstractFileSystem:
        return self.voxel_spacing.fs_overlay

    @property
    def static_is_overlay(self) -> bool:
        return self.fs_static == self.fs_overlay and self.static_path == self.overlay_path

    def _query_static_features(self) -> List[CopickFeaturesFSSpec]:
        if self.static_is_overlay:
            return []

        feat_loc = self.static_path.replace(".zarr", "_")
        paths = self.fs_static.glob(feat_loc + "*_features.zarr") + self.fs_static.glob(feat_loc + "*_features.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_static.isdir(p)]
        feature_types = [n.replace(feat_loc, "").replace("_features.zarr", "") for n in paths]
        # Remove any hidden files?
        feature_types = [ft for ft in feature_types if not ft.startswith(".")]

        feature_types = list(set(feature_types))

        return [
            CopickFeaturesFSSpec(
                tomogram=self,
                meta=CopickFeaturesMeta(
                    tomo_type=self.tomo_type,
                    feature_type=ft,
                ),
                read_only=True,
            )
            for ft in feature_types
        ]

    def _query_overlay_features(self) -> List[CopickFeaturesFSSpec]:
        feat_loc = self.overlay_path.replace(".zarr", "_")
        paths = self.fs_overlay.glob(feat_loc + "*_features.zarr") + self.fs_overlay.glob(feat_loc + "*_features.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_overlay.isdir(p)]
        feature_types = [n.replace(feat_loc, "").replace("_features.zarr", "") for n in paths]
        # Remove any hidden files?
        feature_types = [ft for ft in feature_types if not ft.startswith(".")]

        feature_types = list(set(feature_types))

        return [
            CopickFeaturesFSSpec(
                tomogram=self,
                meta=CopickFeaturesMeta(
                    tomo_type=self.tomo_type,
                    feature_type=ft,
                ),
                read_only=False,
            )
            for ft in feature_types
        ]

    def zarr(self) -> zarr.storage.FSStore:
        """Get the zarr store for the tomogram object.

        Returns:
            zarr.storage.FSStore: The zarr store for the tomogram object.
        """
        if self.read_only:
            fs = self.fs_static
            path = self.static_path
            mode = "r"
            create = False
        else:
            fs = self.fs_overlay
            path = self.overlay_path
            mode = "w"
            create = not fs.exists(path)

        return zarr.storage.FSStore(
            path,
            fs=fs,
            mode=mode,
            key_separator="/",
            dimension_separator="/",
            create=create,
        )


class CopickVoxelSpacingFSSpec(CopickVoxelSpacingOverlay):
    """CopickVoxelSpacing class backed by fsspec storage.

    Attributes:
        static_path (str): The path to the voxel spacing on the static source.
        overlay_path (str): The path to the voxel spacing on the overlay source.
        fs_static (AbstractFileSystem): The filesystem containing the voxel spacing on the static source.
        fs_overlay (AbstractFileSystem): The filesystem containing the voxel spacing on the overlay source.
        static_is_overlay (bool): Whether the static and overlay sources are the same.

    """

    run: "CopickRunFSSpec"

    def _tomogram_factory(self) -> Tuple[Type[CopickTomogramFSSpec], Type[CopickTomogramMeta]]:
        return CopickTomogramFSSpec, CopickTomogramMeta

    @property
    def static_path(self) -> str:
        return f"{self.run.static_path}/VoxelSpacing{self.voxel_size:.3f}"

    @property
    def overlay_path(self) -> str:
        return f"{self.run.overlay_path}/VoxelSpacing{self.voxel_size:.3f}"

    @property
    def fs_static(self) -> AbstractFileSystem:
        return self.run.fs_static

    @property
    def fs_overlay(self) -> AbstractFileSystem:
        return self.run.fs_overlay

    @property
    def static_is_overlay(self) -> bool:
        return self.fs_static == self.fs_overlay and self.static_path == self.overlay_path

    def _query_static_tomograms(self) -> List[CopickTomogramFSSpec]:
        if self.static_is_overlay:
            return []

        tomo_loc = f"{self.static_path}/"
        paths = self.fs_static.glob(tomo_loc + "*.zarr") + self.fs_static.glob(tomo_loc + "*.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_static.isdir(p)]
        tomo_types = [n.replace(tomo_loc, "").replace(".zarr", "") for n in paths]
        tomo_types = [t for t in tomo_types if "features" not in t]
        # Remove any hidden files?
        tomo_types = [tt for tt in tomo_types if not tt.startswith(".")]

        tomo_types = list(set(tomo_types))

        return [
            CopickTomogramFSSpec(
                voxel_spacing=self,
                meta=CopickTomogramMeta(tomo_type=tt),
                read_only=True,
            )
            for tt in tomo_types
        ]

    def _query_overlay_tomograms(self) -> List[CopickTomogramFSSpec]:
        tomo_loc = f"{self.overlay_path}/"
        paths = self.fs_overlay.glob(tomo_loc + "*.zarr") + self.fs_overlay.glob(tomo_loc + "*.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_overlay.isdir(p)]
        tomo_types = [n.replace(tomo_loc, "").replace(".zarr", "") for n in paths]
        tomo_types = [t for t in tomo_types if "features" not in t]
        # Remove any hidden files?
        tomo_types = [tt for tt in tomo_types if not tt.startswith(".")]

        tomo_types = list(set(tomo_types))

        return [
            CopickTomogramFSSpec(
                voxel_spacing=self,
                meta=CopickTomogramMeta(tomo_type=tt),
                read_only=False,
            )
            for tt in tomo_types
        ]

    def ensure(self, create: bool = False) -> bool:
        """Checks if the voxel spacing record exists in the static or overlay directory, optionally creating it in the
        overlay filesystem if it does not.

        Args:
            create: Whether to create the voxel spacing record if it does not exist.

        Returns:
            bool: True if the voxel spacing record exists, False otherwise.
        """
        if self.static_is_overlay:
            exists = self.fs_overlay.exists(self.overlay_path)
        else:
            exists = self.fs_static.exists(self.static_path) or self.fs_overlay.exists(self.overlay_path)

        if not exists and create:
            self.fs_overlay.makedirs(self.overlay_path, exist_ok=True)
            # TODO: Write metadata
            with self.fs_overlay.open(self.overlay_path + "/.meta", "w") as f:
                f.write("meta")  # Touch the file
            return True
        else:
            return exists


class CopickRunFSSpec(CopickRunOverlay):
    """CopickRun class backed by fsspec storage.

    Attributes:
        static_path (str): The path to the run on the static source.
        overlay_path (str): The path to the run on the overlay source.
        fs_static (AbstractFileSystem): The filesystem containing the run on the static source.
        fs_overlay (AbstractFileSystem): The filesystem containing the run on the overlay source.
        static_is_overlay (bool): Whether the static and overlay sources are the same.
    """

    root: "CopickRootFSSpec"

    def _voxel_spacing_factory(self) -> Tuple[Type[CopickVoxelSpacingFSSpec], Type["CopickVoxelSpacingMeta"]]:
        return CopickVoxelSpacingFSSpec, CopickVoxelSpacingMeta

    def _picks_factory(self) -> Type[CopickPicksFSSpec]:
        return CopickPicksFSSpec

    def _mesh_factory(self) -> Tuple[Type[CopickMeshFSSpec], Type[CopickMeshMeta]]:
        return CopickMeshFSSpec, CopickMeshMeta

    def _segmentation_factory(self) -> Tuple[Type[CopickSegmentationFSSpec], Type[CopickSegmentationMeta]]:
        return CopickSegmentationFSSpec, CopickSegmentationMeta

    @property
    def static_path(self) -> str:
        return f"{self.root.root_static}/ExperimentRuns/{self.name}"

    @property
    def overlay_path(self) -> str:
        return f"{self.root.root_overlay}/ExperimentRuns/{self.name}"

    @property
    def fs_static(self) -> AbstractFileSystem:
        return self.root.fs_static

    @property
    def fs_overlay(self) -> AbstractFileSystem:
        return self.root.fs_overlay

    @property
    def static_is_overlay(self) -> bool:
        return self.fs_static == self.fs_overlay and self.static_path == self.overlay_path

    def _query_static_voxel_spacings(self) -> List[CopickVoxelSpacingFSSpec]:
        if self.static_is_overlay:
            return []

        static_vs_loc = f"{self.static_path}/VoxelSpacing"
        spaths = set(self.fs_static.glob(static_vs_loc + "*") + self.fs_static.glob(static_vs_loc + "*/"))
        spaths = [p.rstrip("/") for p in spaths]
        spacings = [float(p.replace(f"{static_vs_loc}", "")) for p in spaths]

        return [
            CopickVoxelSpacingFSSpec(
                meta=CopickVoxelSpacingMeta(voxel_size=s),
                run=self,
            )
            for s in spacings
        ]

    def _query_overlay_voxel_spacings(self) -> List[CopickVoxelSpacingFSSpec]:
        overlay_vs_loc = f"{self.overlay_path}/VoxelSpacing"
        opaths = set(self.fs_overlay.glob(overlay_vs_loc + "*") + self.fs_overlay.glob(overlay_vs_loc + "*/"))
        opaths = [p.rstrip("/") for p in opaths]
        spacings = [float(p.replace(f"{overlay_vs_loc}", "")) for p in opaths]

        return [
            CopickVoxelSpacingFSSpec(
                meta=CopickVoxelSpacingMeta(voxel_size=s),
                run=self,
            )
            for s in spacings
        ]

    def _query_static_picks(self) -> List[CopickPicksFSSpec]:
        if self.static_is_overlay:
            return []

        pick_loc = f"{self.static_path}/Picks/"
        paths = self.fs_static.glob(pick_loc + "*.json")
        names = [n.replace(pick_loc, "").replace(".json", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        users = [n.split("_")[0] for n in names]
        sessions = [n.split("_")[1] for n in names]
        objects = [n.split("_")[2] for n in names]

        # zip(strict=True) (replace once python 3.9 is EOL)
        assert len(users) == len(sessions) == len(objects)

        return [
            CopickPicksFSSpec(
                run=self,
                file=CopickPicksFile(
                    pickable_object_name=o,
                    user_id=u,
                    session_id=s,
                ),
                read_only=True,
            )
            for u, s, o in zip(users, sessions, objects)  # , strict=True)
        ]

    def _query_overlay_picks(self) -> List[CopickPicksFSSpec]:
        pick_loc = f"{self.overlay_path}/Picks/"
        paths = self.fs_overlay.glob(pick_loc + "*.json")
        names = [n.replace(pick_loc, "").replace(".json", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        users = [n.split("_")[0] for n in names]
        sessions = [n.split("_")[1] for n in names]
        objects = [n.split("_")[2] for n in names]

        # zip(strict=True) (replace once python 3.9 is EOL)
        assert len(users) == len(sessions) == len(objects)

        return [
            CopickPicksFSSpec(
                run=self,
                file=CopickPicksFile(
                    pickable_object_name=o,
                    user_id=u,
                    session_id=s,
                ),
                read_only=False,
            )
            for u, s, o in zip(users, sessions, objects)  # , strict=True)
        ]

    def _query_static_meshes(self) -> List[CopickMeshFSSpec]:
        if self.static_is_overlay:
            return []

        mesh_loc = f"{self.static_path}/Meshes/"
        paths = self.fs_static.glob(mesh_loc + "*.glb")
        names = [n.replace(mesh_loc, "").replace(".glb", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        users = [n.split("_")[0] for n in names]
        sessions = [n.split("_")[1] for n in names]
        objects = [n.split("_")[2] for n in names]

        # zip(strict=True) (replace once python 3.9 is EOL)
        assert len(users) == len(sessions) == len(objects)

        return [
            CopickMeshFSSpec(
                run=self,
                meta=CopickMeshMeta(
                    pickable_object_name=o,
                    user_id=u,
                    session_id=s,
                ),
                read_only=True,
            )
            for u, s, o in zip(users, sessions, objects)  # , strict=True)
        ]

    def _query_overlay_meshes(self) -> List[CopickMeshFSSpec]:
        mesh_loc = f"{self.overlay_path}/Meshes/"
        paths = self.fs_overlay.glob(mesh_loc + "*.glb")
        names = [n.replace(mesh_loc, "").replace(".glb", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        users = [n.split("_")[0] for n in names]
        sessions = [n.split("_")[1] for n in names]
        objects = [n.split("_")[2] for n in names]

        # zip(strict=True) (replace once python 3.9 is EOL)
        assert len(users) == len(sessions) == len(objects)

        return [
            CopickMeshFSSpec(
                run=self,
                meta=CopickMeshMeta(
                    pickable_object_name=o,
                    user_id=u,
                    session_id=s,
                ),
                read_only=False,
            )
            for u, s, o in zip(users, sessions, objects)  # , strict=True)
        ]

    def _query_static_segmentations(self) -> List[CopickSegmentationFSSpec]:
        if self.static_is_overlay:
            return []

        seg_loc = f"{self.static_path}/Segmentations/"
        paths = self.fs_static.glob(seg_loc + "*.zarr") + self.fs_static.glob(seg_loc + "*.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_static.isdir(p)]
        names = [n.replace(seg_loc, "").replace(".zarr", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        # Deduplicate
        names = list(set(names))

        # multilabel vs single label
        metas = []
        for n in names:
            if "multilabel" in n:
                parts = n.split("_")
                metas.append(
                    CopickSegmentationMeta(
                        is_multilabel=True,
                        voxel_size=float(parts[0]),
                        user_id=parts[1],
                        session_id=parts[2],
                        name=parts[3].replace("-multilabel", ""),
                    ),
                )
            else:
                parts = n.split("_")
                metas.append(
                    CopickSegmentationMeta(
                        is_multilabel=False,
                        voxel_size=float(parts[0]),
                        user_id=parts[1],
                        session_id=parts[2],
                        name=parts[3],
                    ),
                )

        return [
            CopickSegmentationFSSpec(
                run=self,
                meta=m,
                read_only=True,
            )
            for m in metas
        ]

    def _query_overlay_segmentations(self) -> List[CopickSegmentationFSSpec]:
        seg_loc = f"{self.overlay_path}/Segmentations/"
        paths = self.fs_overlay.glob(seg_loc + "*.zarr") + self.fs_overlay.glob(seg_loc + "*.zarr/")
        paths = [p.rstrip("/") for p in paths if self.fs_overlay.isdir(p)]
        names = [n.replace(seg_loc, "").replace(".zarr", "") for n in paths]
        # Remove any hidden files?
        names = [n for n in names if not n.startswith(".")]

        # Deduplicate
        names = list(set(names))

        # multilabel vs single label
        metas = []
        for n in names:
            if "multilabel" in n:
                parts = n.split("_")
                metas.append(
                    CopickSegmentationMeta(
                        is_multilabel=True,
                        voxel_size=float(parts[0]),
                        user_id=parts[1],
                        session_id=parts[2],
                        name=parts[3].replace("-multilabel", ""),
                    ),
                )
            else:
                parts = n.split("_")
                metas.append(
                    CopickSegmentationMeta(
                        is_multilabel=False,
                        voxel_size=float(parts[0]),
                        user_id=parts[1],
                        session_id=parts[2],
                        name=parts[3],
                    ),
                )

        return [
            CopickSegmentationFSSpec(
                run=self,
                meta=m,
                read_only=False,
            )
            for m in metas
        ]

    def ensure(self, create: bool = False) -> bool:
        """Checks if the run record exists in the static or overlay directory, optionally creating it in the overlay
        filesystem if it does not.

        Args:
            create: Whether to create the run record if it does not exist.

        Returns:
            bool: True if the run record exists, False otherwise.
        """

        if self.static_is_overlay:
            exists = self.fs_overlay.exists(self.overlay_path)
        else:
            exists = self.fs_static.exists(self.static_path) or self.fs_overlay.exists(self.overlay_path)

        if not exists and create:
            self.fs_overlay.makedirs(self.overlay_path, exist_ok=True)
            # TODO: Write metadata
            with self.fs_overlay.open(self.overlay_path + "/.meta", "w") as f:
                f.write("meta")  # Touch the file
            return True
        else:
            return exists


class CopickObjectFSSpec(CopickObjectOverlay):
    """CopickObject class backed by fsspec storage.

    Attributes:
        path (str): The path to the object file.
        fs (AbstractFileSystem): The filesystem containing the object file.
    """

    root: "CopickRootFSSpec"

    @property
    def path(self) -> str:
        return f"{self.root.root_static}/Objects/{self.name}.zarr"

    @property
    def fs(self) -> AbstractFileSystem:
        return self.root.fs_static

    def zarr(self) -> Union[None, zarr.storage.FSStore]:
        """Get the zarr store for the object.

        Returns:
            Union[None, zarr.storage.FSStore]: The zarr store for the object, or None if the object is not a particle.
        """
        if not self.is_particle:
            return None

        if not self.fs.exists(self.path):
            return None

        if self.read_only:
            mode = "r"
            create = False
        else:
            mode = "w"
            create = not self.fs.exists(self.path)

        return zarr.storage.FSStore(
            self.path,
            fs=self.fs,
            mode=mode,
            key_separator="/",
            dimension_separator="/",
            create=create,
        )


class CopickRootFSSpec(CopickRoot):
    """CopickRoot class backed by fspec storage.

    Attributes:
        fs_overlay (AbstractFileSystem): The filesystem for the overlay storage.
        fs_static (Optional[AbstractFileSystem]): The filesystem for the static storage.
        root_overlay (str): The root path for the overlay storage.
        root_static (Optional[str]): The root path for the static storage.
    """

    def __init__(self, config: CopickConfigFSSpec):
        """
        Args:
            config: Copick configuration for fsspec-based storage.
        """
        super().__init__(config)

        self.fs_overlay: AbstractFileSystem = fsspec.core.url_to_fs(config.overlay_root, **config.overlay_fs_args)[0]
        self.fs_static: Optional[AbstractFileSystem] = None

        self.root_overlay: str = self.fs_overlay._strip_protocol(config.overlay_root).rstrip("/")
        self.root_static: Optional[str] = None

        if config.static_root is None:
            self.fs_static = self.fs_overlay
            self.root_static = self.fs_static._strip_protocol(config.overlay_root).rstrip("/")
        else:
            self.fs_static = fsspec.core.url_to_fs(config.static_root, **config.static_fs_args)[0]
            self.root_static = self.fs_static._strip_protocol(config.static_root).rstrip("/")

    @property
    def static_is_overlay(self) -> bool:
        return self.fs_static == self.fs_overlay and self.root_static == self.root_overlay

    @classmethod
    def from_file(cls, path: str) -> "CopickRootFSSpec":
        """Initialize a CopickRootFSSpec from a configuration file on disk.

        Args:
            path: Path to the configuration file on disk.

        Returns:
            CopickRootFSSpec: The initialized CopickRootFSSpec object.
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls(CopickConfigFSSpec(**data))

    def _run_factory(self) -> Tuple[Type[CopickRunFSSpec], Type["CopickRunMeta"]]:
        return CopickRunFSSpec, CopickRunMeta

    def _object_factory(self) -> Tuple[Type[CopickObjectFSSpec], Type[PickableObject]]:
        return CopickObjectFSSpec, PickableObject

    @staticmethod
    def _query_names(fs, root) -> List[str]:
        # Query location
        run_dir = f"{root}/ExperimentRuns/"
        paths = fs.glob(run_dir + "**", maxdepth=1, detail=True)
        names = [p.rstrip("/").replace(run_dir, "") for p, details in paths.items() if details["type"] == "directory"]

        # Remove any hidden files
        names = [n for n in names if not n.startswith(".") and n != f"{root}/ExperimentRuns"]

        return names

    def _query_static_names(self) -> List[str]:
        return self._query_names(self.fs_static, self.root_static)

    def _query_overlay_names(self) -> List[str]:
        return self._query_names(self.fs_overlay, self.root_overlay)

    def query(self) -> List[CopickRunFSSpec]:
        # Query filesystems in parallel
        if self.static_is_overlay:
            tasks = [self._query_overlay_names]
        else:
            tasks = [self._query_static_names, self._query_overlay_names]

        names = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                names += future.result()

        # Deduplicate
        names = sorted(set(names))

        # Create objects
        runs = []
        for n in names:
            rm = CopickRunMeta(name=n)
            runs.append(CopickRunFSSpec(root=self, meta=rm))

        return runs
