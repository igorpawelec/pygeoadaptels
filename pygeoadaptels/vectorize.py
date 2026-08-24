"""
pygeoadaptels — Vectorization of adaptel labels to polygons.

Uses only rasterio.features.shapes + fiona for polygon export.
No geopandas or shapely required (following Netzel's recommendation).

Based on:
    https://github.com/rasterio/rasterio/blob/main/examples/rasterio_polygonize.py
    https://gis.stackexchange.com/questions/417383/how-to-apply-gdal-polygonize
"""

import os
import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _ensure_deps():
    """Lazy import rasterio.features.shapes and fiona."""
    try:
        from rasterio.features import shapes
    except ImportError as e:
        raise ImportError(
            "rasterio is required for vectorization.\n"
            "Install:  conda install -c conda-forge rasterio"
        ) from e
    try:
        import fiona
    except ImportError as e:
        raise ImportError(
            "fiona is required for writing vector files.\n"
            "Install:  conda install -c conda-forge fiona"
        ) from e
    return shapes, fiona


def vectorize_adaptels(labels, transform, crs_wkt,
                       output_path,
                       driver="ESRI Shapefile",
                       nodata=-9999,
                       connectivity=4,
                       compute_area=True,
                       quiet=False):
    """
    Convert adaptel label raster to polygon vector file.

    Uses rasterio.features.shapes for polygonization and fiona for
    writing — no geopandas or shapely needed.

    Parameters
    ----------
    labels : np.ndarray, shape (rows, cols), dtype int32
        Adaptel label raster from create_adaptels() or adaptels_from_array().
    transform : affine.Affine
        Geotransform from the input raster (rasterio src.transform).
    crs_wkt : str
        Coordinate reference system as WKT string (rasterio src.crs.to_wkt()).
    output_path : str or Path
        Output file path. Extension determines format:
        .shp → ESRI Shapefile (default), .gpkg → GeoPackage, .geojson → GeoJSON.
        If no recognized extension, uses `driver` parameter.
    driver : str, optional
        Fiona driver override. Default "ESRI Shapefile".
    nodata : int, optional
        Nodata value in labels array. Default -9999.
    connectivity : int, optional
        Pixel connectivity for polygonization: 4 or 8. Default 4.
    compute_area : bool, optional
        If True, compute polygon area (m²) and perimeter (m) from
        pixel counts. Default True.

    Returns
    -------
    n_polygons : int
        Number of polygons written.

    Examples
    --------
    >>> from pygeoadaptels import create_adaptels
    >>> from pygeoadaptels.vectorize import vectorize_adaptels
    >>> import rasterio
    >>>
    >>> labels, n = create_adaptels('input.tif', threshold=60.0)
    >>> with rasterio.open('input.tif') as src:
    ...     vectorize_adaptels(labels, src.transform, src.crs.to_wkt(),
    ...                        'adaptels.shp')
    """
    shapes, fiona = _ensure_deps()
    output_path = str(output_path)

    # Auto-detect driver from extension
    ext = os.path.splitext(output_path)[1].lower()
    ext_drivers = {".shp": "ESRI Shapefile", ".gpkg": "GPKG", ".geojson": "GeoJSON"}
    if ext in ext_drivers:
        driver = ext_drivers[ext]

    # Prepare data
    data = labels.astype(np.int32)
    mask = (data != int(nodata)) & (data >= 0)

    # Pixel area for attribute computation
    pixel_w = abs(transform.a)
    pixel_h = abs(transform.e)
    pixel_area = pixel_w * pixel_h

    # Schema
    props = {'adaptel_id': 'int'}
    if compute_area:
        props['area_m2'] = 'float'
        props['perimeter'] = 'float'

    schema = {
        'geometry': 'Polygon',
        'properties': props,
    }

    # Precompute pixel counts per adaptel (one pass, O(n_pixels))
    if compute_area:
        valid = data[mask]
        max_id = int(valid.max()) + 1 if valid.size > 0 else 1
        pixel_counts = np.bincount(valid, minlength=max_id)

    # Estimate polygon count for progress bar
    n_unique = int(len(np.unique(data[mask])))

    # Polygonize + write
    n_polygons = 0
    poly_iter = shapes(data, mask=mask, transform=transform,
                       connectivity=connectivity)
    if _HAS_TQDM and not quiet:
        poly_iter = _tqdm(poly_iter, desc="Vectorizing", unit="poly",
                          total=n_unique)

    with fiona.open(
        output_path,
        'w',
        driver=driver,
        crs_wkt=crs_wkt,
        schema=schema
    ) as dst:
        for geom, value in poly_iter:
            adaptel_id = int(value)
            if adaptel_id < 0:
                continue

            feature = {
                'geometry': geom,
                'properties': {'adaptel_id': adaptel_id},
            }

            if compute_area:
                n_pixels = int(pixel_counts[adaptel_id])
                area = n_pixels * pixel_area
                perimeter = 2.0 * np.sqrt(np.pi * area)
                feature['properties']['area_m2'] = round(area, 2)
                feature['properties']['perimeter'] = round(perimeter, 2)

            dst.write(feature)
            n_polygons += 1

    return n_polygons


def vectorize_from_file(input_raster, output_path,
                        driver="ESRI Shapefile",
                        nodata=None, connectivity=4,
                        compute_area=True, quiet=False):
    """
    Convenience wrapper: read adaptel raster from file and vectorize.

    Parameters
    ----------
    input_raster : str or Path
        Path to adaptel GeoTIFF (output of create_adaptels).
    output_path : str or Path
        Output vector file path.
    driver : str
        Fiona driver. Default "ESRI Shapefile".
    nodata : int, optional
        Override nodata. If None, reads from raster metadata.
    connectivity : int
        Pixel connectivity (4 or 8).
    compute_area : bool
        Compute area/perimeter attributes.

    Returns
    -------
    n_polygons : int

    Examples
    --------
    >>> from pygeoadaptels.vectorize import vectorize_from_file
    >>> n = vectorize_from_file('adaptels.tif', 'adaptels.shp')
    >>> print(f"Wrote {n} polygons")
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio required. Install: conda install -c conda-forge rasterio") from e

    with rasterio.open(str(input_raster)) as src:
        data = src.read(1).astype(np.int32)
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
        if nodata is None:
            nodata = src.nodata if src.nodata is not None else -9999

    return vectorize_adaptels(
        data, transform, crs_wkt, output_path,
        driver=driver, nodata=int(nodata),
        connectivity=connectivity, compute_area=compute_area,
        quiet=quiet
    )
