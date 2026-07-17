"""
Raster I/O using rasterio.

Handles reading multi-band GeoTIFF files and writing results.
"""

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def read_raster(filepath):
    """
    Read a raster file and return layers, mask, and metadata.
    
    Parameters
    ----------
    filepath : str or list of str
        Path(s) to input raster file(s). If a list, all files must
        have the same extent, resolution, and projection.
    
    Returns
    -------
    layers : np.ndarray, shape (n_layers, rows*cols), dtype float64
        Flattened pixel values for each band/layer.
    mask : np.ndarray, shape (rows*cols,), dtype uint8
        1 = nodata, 0 = valid pixel.
    meta : dict
        Rasterio metadata for writing output.
    cols : int
    rows : int
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for raster I/O. "
            "Install with: conda install -c conda-forge rasterio"
        )
    
    if isinstance(filepath, str):
        filepath = [filepath]
    
    all_bands = []
    meta = None
    cols = None
    rows = None
    nodata_val = None
    
    for fpath in filepath:
        with rasterio.open(fpath) as src:
            if meta is None:
                meta = src.meta.copy()
                cols = src.width
                rows = src.height
                nodata_val = src.nodata
            else:
                # Verify compatibility
                if src.width != cols or src.height != rows:
                    raise ValueError(
                        f"File {fpath} has different dimensions "
                        f"({src.width}x{src.height}) than first file "
                        f"({cols}x{rows})"
                    )
            
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx).astype(np.float64)
                all_bands.append(data.ravel())
                
                # Update nodata from this band if not set
                if nodata_val is None:
                    nodata_val = src.nodata
    
    n_layers = len(all_bands)
    size = rows * cols
    
    # Stack layers: shape (n_layers, size)
    layers = np.array(all_bands, dtype=np.float64)
    
    # Create mask: 1=nodata, 0=valid
    mask = np.zeros(size, dtype=np.uint8)
    if nodata_val is not None:
        for i in range(n_layers):
            mask[np.isnan(layers[i]) | (layers[i] == nodata_val)] = 1
    else:
        for i in range(n_layers):
            mask[np.isnan(layers[i])] = 1
    
    return layers, mask, meta, cols, rows


def write_raster(filepath, labels, meta, cols, rows, nodata=-9999, 
                 compress='deflate'):
    """
    Write adaptel labels to a GeoTIFF file.
    
    Parameters
    ----------
    filepath : str
        Output file path.
    labels : np.ndarray, shape (rows*cols,), dtype int32
        Adaptel labels.
    meta : dict
        Rasterio metadata from input file.
    cols, rows : int
        Raster dimensions.
    nodata : int
        Nodata value for output.
    compress : str
        Compression algorithm.
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for raster I/O. "
            "Install with: conda install -c conda-forge rasterio"
        )
    
    out_meta = meta.copy()
    out_meta.update({
        'dtype': 'int32',
        'count': 1,
        'nodata': nodata,
        'compress': compress,
    })
    
    data_2d = labels.reshape(rows, cols).astype(np.int32)
    
    with rasterio.open(filepath, 'w', **out_meta) as dst:
        dst.write(data_2d, 1)


def normalize_layers(layers, mask):
    """
    Normalize each layer to [0, 1] range, ignoring nodata pixels.
    
    Parameters
    ----------
    layers : np.ndarray, shape (n_layers, n_pixels)
    mask : np.ndarray, shape (n_pixels,), 1=nodata
    
    Returns
    -------
    layers : normalized copy
    """
    layers = layers.copy()
    valid = mask == 0
    
    for i in range(layers.shape[0]):
        band = layers[i]
        valid_vals = band[valid]
        if len(valid_vals) == 0:
            continue
        vmin = valid_vals.min()
        vmax = valid_vals.max()
        if vmin == vmax:
            band[valid] = 0.0
        else:
            band[valid] = (band[valid] - vmin) / (vmax - vmin)
    
    return layers
