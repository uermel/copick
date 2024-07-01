from typing import List, Optional

from trimesh.parent import Geometry

from copick.models import (
    CopickFeatures,
    CopickFeaturesMeta,
    CopickMesh,
    CopickMeshMeta,
    CopickObject,
    CopickPicks,
    CopickPicksFile,
    CopickRoot,
    CopickRun,
    CopickSegmentation,
    CopickSegmentationMeta,
    CopickTomogram,
    CopickTomogramMeta,
    CopickVoxelSpacing,
    PickableObject,
)


class CopickPicksOverlay(CopickPicks):
    """CopickPicks class that keeps track of whether the picks are read-only.

    Attributes:
        read_only (bool): Whether the picks are read-only.
    """

    def __init__(self, run: CopickRun, file: CopickPicksFile, read_only: bool = False):
        super().__init__(run, file)
        self.read_only = read_only

    def store(self):
        """Store the picks, making sure the source is writable."""
        if self.read_only:
            raise PermissionError("Cannot store picks in a read-only source.")
        self._store()


class CopickMeshOverlay(CopickMesh):
    """CopickMesh class that keeps track of whether the mesh is read-only.

    Attributes:
        read_only (bool): Whether the mesh is read-only.
    """

    def __init__(self, run: CopickRun, meta: CopickMeshMeta, mesh: Optional[Geometry] = None, read_only: bool = False):
        super().__init__(run, meta, mesh)
        self.read_only = read_only

    def store(self):
        """Store the mesh, making sure the source is writable."""
        if self.read_only:
            raise PermissionError("Cannot store mesh in a read-only source.")
        self._store()


class CopickSegmentationOverlay(CopickSegmentation):
    """CopickSegmentation class that keeps track of whether the segmentation is read-only.

    Attributes:
        read_only (bool): Whether the segmentation is read-only.
    """

    def __init__(self, run: CopickRun, meta: CopickSegmentationMeta, read_only: bool = False):
        super().__init__(run, meta)
        self.read_only = read_only


class CopickObjectOverlay(CopickObject):
    """CopickObject class that keeps track of whether the object is read-only.

    Attributes:
        read_only (bool): Whether the object is read-only.
    """

    def __init__(self, root: CopickRoot, meta: PickableObject, read_only: bool = True):
        super().__init__(root, meta)
        self.read_only = read_only


class CopickFeaturesOverlay(CopickFeatures):
    """CopickFeatures class that keeps track of whether the features are read-only.

    Attributes:
        read_only (bool): Whether the features are read-only.
    """

    def __init__(self, tomogram: CopickTomogram, meta: CopickFeaturesMeta, read_only: bool = False):
        super().__init__(tomogram, meta)
        self.read_only = read_only


class CopickTomogramOverlay(CopickTomogram):
    """CopickTomogram class that keeps track of whether the tomogram is read-only and queries two different storage
    locations for tomograms. The first location is read-only (static) and the second location is writable (overlay).

    Attributes:
        read_only (bool): Whether the tomogram is read-only.
    """

    def __init__(self, voxel_spacing: CopickVoxelSpacing, meta: CopickTomogramMeta, read_only: bool = False, **kwargs):
        super().__init__(voxel_spacing, meta, **kwargs)
        self.read_only = read_only

    def _query_static_features(self) -> List[CopickFeaturesOverlay]:
        """Override to query the static source for the features. All returned features must be read-only.

        Returns:
            List[CopickFeaturesOverlay]: List of read-only features.
        """
        raise NotImplementedError("CopickTomogramOverlay must implement _query_static_features method.")

    def _query_overlay_features(self) -> List[CopickFeaturesOverlay]:
        """Override to query the overlay source for the features.

        Returns:
            List[CopickFeaturesOverlay]: List of writable features.
        """
        raise NotImplementedError("CopickTomogramOverlay must implement _query_overlay_features method.")

    def query_features(self) -> List[CopickFeaturesOverlay]:
        """Query all features.

        Returns:
            List[CopickFeaturesOverlay]: List of features from both sources.
        """
        static = self._query_static_features()
        overlay = self._query_overlay_features()

        for f in static:
            assert f.read_only, "Features from static source must be read-only."

        return static + overlay


class CopickVoxelSpacingOverlay(CopickVoxelSpacing):
    """CopickVoxelSpacing class that queries two different storage locations for voxel spacings. The first location is
    read-only (static) and the second location is writable (overlay).
    """

    def _query_static_tomograms(self) -> List[CopickTomogramOverlay]:
        """Override to query the static source for the tomograms. All returned tomograms must be read-only.

        Returns:
            List[CopickTomogramOverlay]: List of read-only tomograms.
        """
        raise NotImplementedError("CopickVoxelSpacingOverlay must implement _query_static_tomograms method.")

    def _query_overlay_tomograms(self) -> List[CopickTomogramOverlay]:
        """Override to query the overlay source for the tomograms.

        Returns:
            List[CopickTomogramOverlay]: List of writable tomograms.
        """
        raise NotImplementedError("CopickVoxelSpacingOverlay must implement _query_overlay_tomograms method.")

    def query_tomograms(self) -> List[CopickTomogramOverlay]:
        """Query all tomograms.

        Returns:
            List[CopickTomogramOverlay]: List of tomograms from both sources.
        """
        static = self._query_static_tomograms()
        overlay = self._query_overlay_tomograms()

        for t in static:
            assert t.read_only, "Tomograms from static source must be read-only."

        return static + overlay


class CopickRunOverlay(CopickRun):
    """CopickRun class that queries two different storage locations for runs. The first location is read-only (static)
    and the second location is writable (overlay).
    """

    def _query_static_voxel_spacings(self) -> List[CopickVoxelSpacingOverlay]:
        """Override to query the static source for the voxel spacings. All returned voxel spacings must be read-only.

        Returns:
            List[CopickVoxelSpacingOverlay]: List of read-only voxel spacings.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_static_voxel_spacings method.")

    def _query_overlay_voxel_spacings(self) -> List[CopickVoxelSpacingOverlay]:
        """Override to query the overlay source for the voxel spacings.

        Returns:
            List[CopickVoxelSpacingOverlay]: List of writable voxel spacings.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_overlay_voxel_spacings method.")

    def query_voxelspacings(self) -> List[CopickVoxelSpacingOverlay]:
        """Query all voxel spacings.

        Returns:
            List[CopickVoxelSpacingOverlay]: List of voxel spacings from both sources.
        """
        static = self._query_static_voxel_spacings()
        overlay = self._query_overlay_voxel_spacings()

        # Remove overlay voxel spacings that are already in the static source.
        sspacings = [v.voxel_size for v in static]
        overlay = [v for v in overlay if v.voxel_size not in sspacings]

        return static + overlay

    def _query_static_picks(self) -> List[CopickPicksOverlay]:
        """Override to query the static source for the picks. All returned picks must be read-only.

        Returns:
            List[CopickPicksOverlay]: List of read-only picks.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_static_picks method.")

    def _query_overlay_picks(self) -> List[CopickPicksOverlay]:
        """Override to query the overlay source for the picks.

        Returns:
            List[CopickPicksOverlay]: List of writable picks.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_overlay_picks method.")

    def query_picks(self) -> List[CopickPicksOverlay]:
        """Query all picks.

        Returns:
            List[CopickPicksOverlay]: List of picks from both sources.
        """
        static = self._query_static_picks()
        overlay = self._query_overlay_picks()

        for p in static:
            assert p.read_only, "Picks from static source must be read-only."

        return static + overlay

    def _query_static_meshes(self) -> List[CopickMeshOverlay]:
        """Override to query the static source for the meshes. All returned meshes must be read-only.

        Returns:
            List[CopickMeshOverlay]: List of read-only meshes.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_static_meshes method.")

    def _query_overlay_meshes(self) -> List[CopickMeshOverlay]:
        """Override to query the overlay source for the meshes.

        Returns:
            List[CopickMeshOverlay]: List of writable meshes.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_overlay_meshes method.")

    def query_meshes(self) -> List[CopickMeshOverlay]:
        """Query all meshes.

        Returns:
            List[CopickMeshOverlay]: List of meshes from both sources.
        """
        static = self._query_static_meshes()
        overlay = self._query_overlay_meshes()

        for m in static:
            assert m.read_only, "Meshes from static source must be read-only."

        return static + overlay

    def _query_static_segmentations(self) -> List[CopickSegmentationOverlay]:
        """Override to query the static source for the segmentations. All returned segmentations must be read-only.

        Returns:
            List[CopickSegmentationOverlay]: List of read-only segmentations.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_static_segmentations method.")

    def _query_overlay_segmentations(self) -> List[CopickSegmentationOverlay]:
        """Override to query the overlay source for the segmentations.

        Returns:
            List[CopickSegmentationOverlay]: List of writable segmentations.
        """
        raise NotImplementedError("CopickRunOverlay must implement _query_overlay_segmentations method.")

    def query_segmentations(self) -> List[CopickSegmentationOverlay]:
        """Query all segmentations.

        Returns:
            List[CopickSegmentationOverlay]: List of segmentations from both sources.
        """
        static = self._query_static_segmentations()
        overlay = self._query_overlay_segmentations()

        for s in static:
            assert s.read_only, "Segmentations from static source must be read-only."

        return static + overlay
