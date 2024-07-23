from typing import Any, Dict, List, MutableMapping, Tuple

import numpy as np
import psutil
import zarr
from ome_zarr.writer import write_multiscale
from skimage.transform import downscale_local_mean, rescale


def _ome_zarr_axes() -> List[Dict[str, str]]:
    return [
        {
            "name": "z",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "angstrom",
        },
    ]


def _ome_zarr_transforms(voxel_size: float) -> List[Dict[str, Any]]:
    return [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]


def volume_pyramid(
    volume: np.ndarray,
    voxel_size: float,
    levels: int,
    dtype: np.dtype = np.float32,
) -> Dict[float, np.ndarray]:
    """Create a volume pyramid by downscaling with interpolation, maintaining the local mean.

    Args:
        volume: The volume to downsample.
        voxel_size: The voxel size of the input volume.
        levels: The number of levels in the pyramid.
        dtype: The data type of the output arrays.

    Returns:
        A dictionary containing the pyramid with the voxel size as the key.
    """
    pyramid = {voxel_size: volume.astype(dtype)}
    vs = voxel_size

    for _ in range(1, levels):
        array = pyramid[vs]
        vs *= 2
        pyramid[vs] = downscale_local_mean(array, (2, 2, 2)).astype(dtype)

    return pyramid


def segmentation_pyramid(
    segmentation: np.ndarray,
    voxel_size: float,
    levels: int,
    dtype: np.dtype = np.int8,
) -> Dict[float, np.ndarray]:
    """Create an image pyramid by downsampling without interpolation.

    Args:
        segmentation: The segmentation to downsample.
        voxel_size: The voxel size of the input segmentation.
        levels: The number of levels in the pyramid.
        dtype: The data type of the output arrays.

    Returns:
        A dictionary containing the pyramid with the voxel size as the key.
    """
    pyramid = {voxel_size: segmentation.astype(dtype)}
    vs = voxel_size

    for _ in range(1, levels):
        array = pyramid[vs]
        vs *= 2
        pyramid[vs] = (
            rescale(
                array,
                (1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
                anti_aliasing=False,
                preserve_range=True,
                order=0,
            ).astype(dtype),
        )

    return pyramid


def ome_metadata(pyramid: Dict[float, np.ndarray]) -> Dict[str, Any]:
    return {
        "axes": _ome_zarr_axes(),
        "transforms": [_ome_zarr_transforms(voxel_size) for voxel_size in pyramid],
    }


def write_ome_zarr_3d(
    store: MutableMapping,
    pyramid: Dict[float, np.ndarray],
    chunk_size: Tuple[int, ...] = (256, 256, 256),
) -> None:
    """Write a 3D pyramid to an OME-Zarr store.

    Args:
        store: The store to write to.
        pyramid: The pyramid to write.
        chunk_size: The chunk size to use for the Zarr store. Default is (256, 256, 256).
    """
    ome_meta = ome_metadata(pyramid)
    root_group = zarr.group(store=store, overwrite=True)

    write_multiscale(
        list(pyramid.values()),
        group=root_group,
        axes=ome_meta["axes"],
        coordinate_transformations=ome_meta["transforms"],
        storage_options=dict(chunks=chunk_size, overwrite=True),
        compute=True,
        metadata={},
    )


def fits_in_memory(array: zarr.Group, slices: Tuple[slice, ...]) -> Tuple[bool, int, int]:
    """Check if the array fits in memory after slicing.

    Args:
        array: The Zarr array to check.
        slices: The slices to apply to the array.

    Returns:
        A tuple containing:
            - A boolean indicating if the array fits in memory.
            - The number of bytes requested.
            - The number of bytes available.
    """

    num_elem = []
    for dim, sl in zip(array.shape, slices):
        num_elem.append(len(range(*sl.indices(dim))))

    requested = np.prod(np.array(num_elem)) * array.itemsize
    available = psutil.virtual_memory().available
    fits = requested < available

    return fits, requested, available
