# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Shared utilities for segmentation services and refiners.

These functions are used by SAM2, SAM3, and other segmentation model services
to convert masks to polygons and simplify polygon geometry.

Dependencies:
    - shapely: For polygon simplification and geometry operations
    - kwimage: For mask-to-polygon conversion
    - PIL: For image loading
    - numpy: For array operations
"""

from typing import List, Optional, Tuple, Union

import numpy as np


def simplify_polygon_to_max_points(
    polygon: List[List[float]],
    max_points: int = 25,
    min_tolerance: float = 0.1,
    max_tolerance: float = 100.0,
) -> List[List[float]]:
    """
    Simplify a polygon to have at most max_points vertices using Douglas-Peucker algorithm.

    Uses binary search to find the optimal tolerance value that results in
    a polygon with at most max_points vertices while preserving as much detail as possible.

    Args:
        polygon: List of [x, y] coordinate pairs
        max_points: Maximum number of points allowed in output polygon
        min_tolerance: Minimum tolerance for simplification
        max_tolerance: Maximum tolerance for simplification

    Returns:
        Simplified polygon as list of [x, y] coordinate pairs
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    if len(polygon) <= max_points:
        return polygon

    # Create shapely polygon
    try:
        shape = ShapelyPolygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)  # Fix invalid geometries
    except Exception:
        return polygon

    # Binary search to find optimal tolerance
    low = min_tolerance
    high = max_tolerance
    best_result = polygon

    for _ in range(20):  # Max iterations for binary search
        mid = (low + high) / 2
        simplified = shape.simplify(mid, preserve_topology=True)

        if simplified.is_empty:
            high = mid
            continue

        # Handle MultiPolygon: extract largest polygon
        if simplified.geom_type == 'MultiPolygon':
            valid_polys = [g for g in simplified.geoms if g.geom_type == 'Polygon' and not g.is_empty]
            if valid_polys:
                simplified = max(valid_polys, key=lambda p: p.area)
            else:
                high = mid
                continue

        if simplified.geom_type != 'Polygon' or simplified.exterior is None:
            high = mid
            continue

        coords = list(simplified.exterior.coords)
        num_points = len(coords)

        if num_points <= max_points:
            best_result = [[float(x), float(y)] for x, y in coords]
            high = mid  # Try to find a smaller tolerance (more detail)
        else:
            low = mid  # Need more simplification

        # Close enough
        if abs(num_points - max_points) <= 2 and num_points <= max_points:
            break

    return best_result


def adaptive_simplify_polygon(
    polygon: List[List[float]],
    max_points: int = 25,
    min_points: int = 4,
    min_tolerance: float = 0.1,
    max_tolerance: float = 100.0,
) -> List[List[float]]:
    """
    Adaptively simplify a polygon based on its shape complexity.

    Unlike simplify_polygon_to_max_points which always tries to reach max_points,
    this function estimates the appropriate number of points based on shape complexity.
    Simple shapes (like rectangles or circles) will use fewer points, while complex
    shapes will use more points up to the maximum.

    The complexity is estimated using the "compactness" metric:
    compactness = 4 * pi * area / perimeter^2
    - A perfect circle has compactness = 1
    - A square has compactness ≈ 0.785
    - Long thin shapes have low compactness

    We also consider the convex hull deviation to detect shapes with concavities.

    Args:
        polygon: List of [x, y] coordinate pairs
        max_points: Maximum number of points allowed in output polygon
        min_points: Minimum number of points (default 4 for basic shapes)
        min_tolerance: Minimum tolerance for Douglas-Peucker simplification
        max_tolerance: Maximum tolerance for Douglas-Peucker simplification

    Returns:
        Simplified polygon as list of [x, y] coordinate pairs
    """
    import math
    from shapely.geometry import Polygon as ShapelyPolygon

    if len(polygon) <= min_points:
        return polygon

    # Create shapely polygon
    try:
        shape = ShapelyPolygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)  # Fix invalid geometries
        if shape.is_empty:
            return polygon
    except Exception:
        return polygon

    # Calculate shape metrics
    area = shape.area
    perimeter = shape.length

    if perimeter <= 0 or area <= 0:
        return polygon

    # Calculate compactness (circularity): 4 * pi * area / perimeter^2
    # Values close to 1 indicate circular/simple shapes
    # Values close to 0 indicate complex/elongated shapes
    compactness = (4 * math.pi * area) / (perimeter * perimeter)

    # Calculate convex hull ratio: shape_area / convex_hull_area
    # Values close to 1 indicate convex shapes
    # Lower values indicate shapes with concavities
    convex_hull = shape.convex_hull
    hull_area = convex_hull.area if convex_hull.area > 0 else area
    convexity = area / hull_area

    # Calculate the target number of points based on complexity
    # Higher compactness = simpler shape = fewer points needed
    # Lower convexity = more concavities = more points needed

    # Complexity score from 0 (simple) to 1 (complex)
    # - High compactness and high convexity = simple (score near 0)
    # - Low compactness or low convexity = complex (score near 1)
    complexity = 1.0 - (compactness * convexity)

    # Map complexity to target points
    # Simple shapes (complexity ~0): use min_points
    # Complex shapes (complexity ~1): use max_points
    target_points = int(min_points + complexity * (max_points - min_points))
    target_points = max(min_points, min(max_points, target_points))

    # If original polygon already has fewer points than target, return as-is
    if len(polygon) <= target_points:
        return polygon

    # Simplify to target number of points using binary search
    low = min_tolerance
    high = max_tolerance
    best_result = polygon

    for _ in range(20):  # Max iterations for binary search
        mid = (low + high) / 2
        simplified = shape.simplify(mid, preserve_topology=True)

        if simplified.is_empty:
            high = mid
            continue

        # Handle MultiPolygon: extract largest polygon
        if simplified.geom_type == 'MultiPolygon':
            valid_polys = [g for g in simplified.geoms if g.geom_type == 'Polygon' and not g.is_empty]
            if valid_polys:
                simplified = max(valid_polys, key=lambda p: p.area)
            else:
                high = mid
                continue

        if simplified.geom_type != 'Polygon' or simplified.exterior is None:
            high = mid
            continue

        coords = list(simplified.exterior.coords)
        num_points = len(coords)

        if num_points <= target_points:
            best_result = [[float(x), float(y)] for x, y in coords]
            high = mid  # Try to find a smaller tolerance (more detail)
        else:
            low = mid  # Need more simplification

        # Close enough
        if num_points <= target_points and num_points >= min_points:
            break

    return best_result


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from path and return as numpy array (RGB).

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array in RGB format with shape (H, W, 3)
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def kwimage_mask_to_shapely(
    mask: np.ndarray,
    pixels_are: str = 'points',
    origin_convention: str = 'center',
):
    """
    Convert a numpy binary mask to a shapely geometry via kwimage.

    This handles the conversion from numpy mask -> kwimage.Mask -> kwimage.MultiPolygon
    -> shapely geometry with robust error handling for edge cases.

    Args:
        mask: Binary mask as numpy array (H, W) with non-zero values indicating the mask
        pixels_are: How to interpret pixels ('points' or 'areas')
        origin_convention: Origin convention ('center' or 'corner')

    Returns:
        shapely.geometry.base.BaseGeometry: A shapely geometry (usually MultiPolygon or Polygon),
            or None if conversion fails or produces empty geometry.
    """
    import kwimage
    from shapely.geometry import MultiPolygon

    # Convert to kwimage mask and then to multi-polygon
    kw_mask = kwimage.Mask.coerce(mask.astype(np.uint8))
    kw_mpoly = kw_mask.to_multi_polygon(
        pixels_are=pixels_are,
        origin_convention=origin_convention,
    )

    try:
        shape = kw_mpoly.to_shapely()
    except ValueError:
        # Workaround for issues with not enough coordinates for a linear ring
        new_parts = []
        for kw_poly in kw_mpoly.data:
            try:
                new_part = kw_poly.to_shapely()
                new_parts.append(new_part)
            except ValueError:
                pass
        if not new_parts:
            return None
        shape = MultiPolygon(new_parts)

    if shape.is_empty:
        return None

    return shape


def apply_polygon_policies(
    shape,
    hole_policy: str = "remove",
    multipolygon_policy: str = "largest",
):
    """
    Apply hole and multipolygon policies to a shapely geometry.

    This function processes a shapely geometry (typically from mask conversion)
    and applies configurable policies for handling:
    - Multiple disjoint polygons (multipolygon_policy)
    - Holes/interior rings within polygons (hole_policy)

    Args:
        shape: A shapely geometry (Polygon, MultiPolygon, or GeometryCollection)
        hole_policy: How to handle holes in polygons:
            - "remove": Remove interior rings (default)
            - "allow": Keep holes as-is
        multipolygon_policy: How to handle multiple polygons:
            - "largest": Keep only the largest polygon by area (default)
            - "convex_hull": Return the convex hull of all polygons
            - "allow": Keep as-is (returns MultiPolygon)

    Returns:
        shapely.geometry.base.BaseGeometry: Processed geometry, or None if empty/invalid.
            When multipolygon_policy is 'allow', returns MultiPolygon.
            Otherwise returns a single Polygon.
    """
    from shapely.geometry import MultiPolygon, Polygon

    if shape is None or shape.is_empty:
        return None

    # Apply multipolygon policy
    if shape.type == 'MultiPolygon' and len(shape.geoms) > 1:
        if multipolygon_policy == 'convex_hull':
            shape = shape.convex_hull
        elif multipolygon_policy == 'largest':
            valid_polys = [g for g in shape.geoms if g.type == 'Polygon' and not g.is_empty]
            if valid_polys:
                shape = max(valid_polys, key=lambda p: p.area)
            else:
                return None
        # 'allow' keeps as-is

    # Apply hole policy
    if hole_policy == 'remove':
        if shape.type == 'Polygon':
            shape = Polygon(shape.exterior)
        elif shape.type == 'MultiPolygon':
            shape = MultiPolygon([Polygon(p.exterior) for p in shape.geoms])

    if shape.is_empty:
        return None

    return shape


def extract_single_polygon(geom):
    """
    Extract a single Polygon from various shapely geometry types.

    Useful when you need exactly one polygon from a potentially complex geometry.
    For MultiPolygon or GeometryCollection, returns the largest polygon by area.

    Args:
        geom: A shapely geometry object

    Returns:
        shapely.geometry.Polygon or None: The extracted polygon, or None if not possible
    """
    if geom is None or geom.is_empty:
        return None

    if geom.type == 'Polygon':
        return geom
    elif geom.type == 'MultiPolygon':
        if geom.geoms:
            valid_polys = [g for g in geom.geoms if g.type == 'Polygon' and not g.is_empty]
            if valid_polys:
                return max(valid_polys, key=lambda p: p.area)
        return None
    elif geom.type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.type == 'Polygon' and not g.is_empty]
        if polys:
            return max(polys, key=lambda p: p.area)
        return None
    else:
        # For other types (Point, LineString, etc.), return None
        return None


def mask_to_polygon(
    mask: np.ndarray,
    hole_policy: str = "remove",
    multipolygon_policy: str = "largest",
) -> Tuple[List[List[float]], List[float]]:
    """
    Convert binary mask to polygon coordinates.

    Uses kwimage for mask-to-polygon conversion and shapely for geometry cleanup.
    This is a convenience function that combines kwimage_mask_to_shapely,
    apply_polygon_policies, and extract_single_polygon.

    Args:
        mask: Binary mask as numpy array (H, W) with non-zero values indicating the mask
        hole_policy: How to handle holes in polygons:
            - "remove": Remove interior rings (default)
            - "allow": Keep holes as-is
        multipolygon_policy: How to handle multiple polygons:
            - "largest": Keep only the largest polygon by area (default)
            - "convex_hull": Return the convex hull of all polygons
            - "allow": Keep as-is (may cause issues with some consumers)

    Returns:
        Tuple of:
            - polygon: List of [x, y] coordinate pairs
            - bounds: [x_min, y_min, x_max, y_max]
    """
    # Convert mask to shapely geometry
    shape = kwimage_mask_to_shapely(mask)
    if shape is None:
        return [], [0, 0, 0, 0]

    # Apply policies
    shape = apply_polygon_policies(shape, hole_policy, multipolygon_policy)
    if shape is None:
        return [], [0, 0, 0, 0]

    # Extract single polygon
    poly = extract_single_polygon(shape)
    if poly is None:
        return [], [0, 0, 0, 0]

    # Get the exterior coordinates
    if poly.is_empty or poly.exterior is None:
        return [], [0, 0, 0, 0]

    coords = list(poly.exterior.coords)
    polygon = [[float(x), float(y)] for x, y in coords]

    # Calculate bounds
    bounds = list(poly.bounds)  # (minx, miny, maxx, maxy)

    return polygon, bounds


def shapely_to_kwimage_multipolygon(shape):
    """
    Convert a shapely geometry back to kwimage.MultiPolygon.

    Useful when you need to convert back to a mask after applying policies.

    Args:
        shape: A shapely geometry (Polygon or MultiPolygon)

    Returns:
        kwimage.MultiPolygon: The converted multipolygon
    """
    import kwimage

    return kwimage.MultiPolygon.from_shapely(shape)


def shapely_to_mask(
    shape,
    dims: Tuple[int, int],
    pixels_are: str = 'points',
    origin_convention: str = 'center',
) -> np.ndarray:
    """
    Convert a shapely geometry to a binary mask.

    Args:
        shape: A shapely geometry (Polygon or MultiPolygon)
        dims: The (height, width) dimensions of the output mask
        pixels_are: How to interpret pixels ('points' or 'areas')
        origin_convention: Origin convention ('center' or 'corner')

    Returns:
        numpy.ndarray: Binary mask with shape (H, W)
    """
    import kwimage

    kw_mpoly = kwimage.MultiPolygon.from_shapely(shape)
    kw_mask = kw_mpoly.to_mask(
        dims=dims,
        pixels_are=pixels_are,
        origin_convention=origin_convention,
    )
    return kw_mask.data
