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
from viame import image_kernels


def simplify_polygon_to_max_points(
    polygon: List[List[float]],
    max_points: int = 25,
    min_tolerance: float = 0.1,
    max_tolerance: Optional[float] = None,
) -> List[List[float]]:
    """
    Simplify a polygon to have at most max_points vertices using Douglas-Peucker algorithm.

    Uses binary search to find the optimal tolerance value that results in
    a polygon with at most max_points vertices while preserving as much detail as possible.

    Args:
        polygon: List of [x, y] coordinate pairs
        max_points: Maximum number of points allowed in output polygon
        min_tolerance: Minimum tolerance for simplification
        max_tolerance: Maximum tolerance for simplification (auto-computed if None)

    Returns:
        Simplified polygon as list of [x, y] coordinate pairs
    """
    import math
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

    # Compute max_tolerance based on polygon size if not specified
    # Use 10% of the bounding box diagonal as a reasonable max tolerance
    if max_tolerance is None:
        bounds = shape.bounds  # (minx, miny, maxx, maxy)
        diagonal = math.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
        max_tolerance = max(100.0, diagonal * 0.1)

    # Binary search to find optimal tolerance
    low = min_tolerance
    high = max_tolerance
    best_result = polygon

    for _ in range(25):  # Max iterations for binary search
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

    # If we still have too many points (rare edge case), try one more aggressive simplification
    if len(best_result) > max_points:
        simplified = shape.simplify(high * 2, preserve_topology=True)
        if not simplified.is_empty and simplified.geom_type == 'Polygon' and simplified.exterior is not None:
            coords = list(simplified.exterior.coords)
            if len(coords) <= max_points:
                best_result = [[float(x), float(y)] for x, y in coords]

    return best_result


def adaptive_simplify_polygon(
    polygon: List[List[float]],
    max_points: int = 25,
    min_points: int = 4,
    min_tolerance: float = 0.1,
    max_tolerance: Optional[float] = None,
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
        max_tolerance: Maximum tolerance for simplification (auto-computed if None)

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

    # Compute max_tolerance based on polygon size if not specified
    if max_tolerance is None:
        bounds = shape.bounds  # (minx, miny, maxx, maxy)
        diagonal = math.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
        max_tolerance = max(100.0, diagonal * 0.1)

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

    for _ in range(25):  # Max iterations for binary search
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

    # If we still have too many points, try one more aggressive simplification
    if len(best_result) > max_points:
        simplified = shape.simplify(high * 2, preserve_topology=True)
        if not simplified.is_empty and simplified.geom_type == 'Polygon' and simplified.exterior is not None:
            coords = list(simplified.exterior.coords)
            if len(coords) <= max_points:
                best_result = [[float(x), float(y)] for x, y in coords]

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
    if shape.geom_type == 'MultiPolygon' and len(shape.geoms) > 1:
        if multipolygon_policy == 'convex_hull':
            shape = shape.convex_hull
        elif multipolygon_policy == 'largest':
            valid_polys = [g for g in shape.geoms if g.geom_type == 'Polygon' and not g.is_empty]
            if valid_polys:
                shape = max(valid_polys, key=lambda p: p.area)
            else:
                return None
        # 'allow' keeps as-is

    # Apply hole policy
    if hole_policy == 'remove':
        if shape.geom_type == 'Polygon':
            shape = Polygon(shape.exterior)
        elif shape.geom_type == 'MultiPolygon':
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

    if geom.geom_type == 'Polygon':
        return geom
    elif geom.geom_type == 'MultiPolygon':
        if geom.geoms:
            valid_polys = [g for g in geom.geoms if g.geom_type == 'Polygon' and not g.is_empty]
            if valid_polys:
                return max(valid_polys, key=lambda p: p.area)
        return None
    elif geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon' and not g.is_empty]
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


def mask_to_polygons(
    mask: np.ndarray,
    hole_policy: str = "allow",
    multipolygon_policy: str = "allow",
    min_area_fraction: float = 0.01,
    keep_points=None,
) -> Tuple[List[dict], List[float]]:
    """
    Convert binary mask to multiple polygon coordinates with hole support.

    Returns all polygons (optionally with holes) from the mask, unlike
    mask_to_polygon() which returns only a single polygon exterior.

    Each polygon dict has:
        - "exterior": List of [x, y] coordinate pairs for the outer ring
        - "holes": List of hole rings, each a list of [x, y] pairs

    Small disconnected polygons (area < min_area_fraction of the largest)
    are discarded as noise.

    Args:
        mask: Binary mask as numpy array (H, W) with non-zero values indicating the mask
        hole_policy: How to handle holes in polygons:
            - "allow": Keep holes as-is (default)
            - "remove": Remove interior rings
        multipolygon_policy: How to handle multiple polygons:
            - "allow": Keep all polygons (default)
            - "largest": Keep only the largest polygon by area
            - "convex_hull": Return the convex hull of all polygons
        min_area_fraction: Minimum polygon area as fraction of the largest polygon.
        keep_points: [x, y] points (e.g. positive prompts) whose polygons are
            kept however small they are.
            Polygons smaller than this are discarded as noise. (default: 0.01 = 1%)

    Returns:
        Tuple of:
            - polygons: List of dicts with "exterior" and "holes" keys
            - bounds: [x_min, y_min, x_max, y_max] overall bounds
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    # Convert mask to shapely geometry
    shape = kwimage_mask_to_shapely(mask)
    if shape is None:
        return [], [0, 0, 0, 0]

    # Apply policies
    shape = apply_polygon_policies(shape, hole_policy, multipolygon_policy)
    if shape is None:
        return [], [0, 0, 0, 0]

    # Collect individual shapely polygons
    shapely_polys: list = []
    if shape.geom_type == 'Polygon':
        if not shape.is_empty and shape.exterior is not None:
            shapely_polys.append(shape)
    elif shape.geom_type == 'MultiPolygon':
        for geom in shape.geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty and geom.exterior is not None:
                shapely_polys.append(geom)

    if not shapely_polys:
        return [], [0, 0, 0, 0]

    # Filter small polygons by area relative to the largest, keeping any that
    # hold a point the caller insists on.
    if len(shapely_polys) > 1 and min_area_fraction > 0:
        from shapely.geometry import Point as ShapelyPoint

        max_area = max(p.area for p in shapely_polys)
        threshold = max_area * min_area_fraction
        anchors = [ShapelyPoint(x, y) for x, y in (keep_points or [])]
        shapely_polys = [
            p for p in shapely_polys
            if p.area >= threshold or any(p.intersects(a) for a in anchors)
        ]

    if not shapely_polys:
        return [], [0, 0, 0, 0]

    # Sort by area descending (largest first = primary polygon)
    shapely_polys.sort(key=lambda p: p.area, reverse=True)

    # Extract coordinates
    polygons = []
    for poly in shapely_polys:
        exterior = [[float(x), float(y)] for x, y in poly.exterior.coords]
        holes = []
        for interior in poly.interiors:
            hole = [[float(x), float(y)] for x, y in interior.coords]
            holes.append(hole)
        polygons.append({"exterior": exterior, "holes": holes})

    # Compute overall bounds from all polygons
    from shapely.ops import unary_union
    combined = unary_union(shapely_polys)
    overall_bounds = list(combined.bounds)  # (minx, miny, maxx, maxy)

    return polygons, overall_bounds


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


_KEYPOINT_ALGO = None


def polygon_keypoint_algo():
    """The add_keypoints_from_mask vital algorithm, configured as the
    measurement and keypoint pipelines configure it (hull_extremes,
    clip_to_mask), so interactively derived head/tail agree with batch."""
    global _KEYPOINT_ALGO
    if _KEYPOINT_ALGO is None:
        from kwiver.vital.algo import RefineDetections
        algo = RefineDetections.create("add_keypoints_from_mask")
        cfg = algo.get_configuration()
        cfg.set_value("method", "hull_extremes")
        cfg.set_value("clip_to_mask", "true")
        algo.set_configuration(cfg)
        _KEYPOINT_ALGO = algo
    return _KEYPOINT_ALGO


def polygons_to_mask(polygons):
    """Rasterize all components and holes of one fish in a shared image crop."""
    components = []
    for polygon in polygons:
        rings = [np.asarray(ring, dtype=np.float64) for ring in
                 [polygon["exterior"], *polygon.get("holes", [])]]
        if any(ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] != 2
               or not np.isfinite(ring).all() for ring in rings):
            raise ValueError("Polygon rings require at least three finite x/y points")
        components.append(rings)
    if not components:
        return None
    pts = np.concatenate([rings[0] for rings in components])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    w, h = int(x1 - x0), int(y1 - y0)
    if w <= 0 or h <= 0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    for rings in components:
        component = np.zeros_like(mask)
        shifted = [np.asarray(ring, dtype=np.float64) - [x0, y0]
                   for ring in rings]
        image_kernels.fill_polygon(component, shifted[0], 255)
        for hole in shifted[1:]:
            image_kernels.fill_polygon(component, hole, 0)
        # Union separately so a hole cannot erase another polygon's foreground.
        np.maximum(mask, component, out=mask)
    return mask, (int(x0), int(y0), int(x1), int(y1))


def polygon_to_keypoints(polygon) -> Optional[Tuple[List[float], List[float]]]:
    """Backward-compatible single-polygon head/tail extraction."""
    return polygons_to_keypoints([{"exterior": polygon, "holes": []}])


def polygons_to_keypoints(polygons) -> Optional[Tuple[List[float], List[float]]]:
    """Derive one head/tail pair from all components of a fish's mask."""
    from kwiver.vital.types import (
        DetectedObject, DetectedObjectSet, BoundingBoxD, ImageContainer, Image)

    rasterized = polygons_to_mask(polygons)
    if rasterized is None:
        return None
    mask, (x0, y0, x1, y1) = rasterized
    h, w = mask.shape
    det = DetectedObject(
        BoundingBoxD(float(x0), float(y0), float(x1), float(y1)),
        1.0, None, ImageContainer(Image(mask)))
    dummy = ImageContainer(Image(np.zeros((h, w, 3), dtype=np.uint8)))

    dets = list(polygon_keypoint_algo().refine(dummy, DetectedObjectSet([det])))
    if not dets:
        return None
    kps = dets[0].keypoints
    if 'head' not in kps or 'tail' not in kps:
        return None
    head, tail = kps['head'].value, kps['tail'].value
    return ([float(head[0]), float(head[1])], [float(tail[0]), float(tail[1])])


def polyline_length(line) -> float:
    pts = np.asarray(line, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def mask_to_line_scale(bounds, line) -> Optional[float]:
    """A mask's longer box side relative to the length of the head/tail line
    it was prompted from; the line spans the object, so this sits near 1."""
    length = polyline_length(line)
    if length <= 0 or bounds is None:
        return None
    return float(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / length


def mask_oversized_for_line(bounds, line, max_ratio: float = 2.5) -> bool:
    """True when the model latched onto something far larger than the line."""
    scale = mask_to_line_scale(bounds, line)
    return scale is not None and scale > max_ratio


def mask_undersized_for_line(bounds, line, min_ratio: float = 0.5) -> bool:
    scale = mask_to_line_scale(bounds, line)
    return scale is not None and scale < min_ratio


def line_background_points(
    line, image_size: Tuple[int, int],
    end_ratio: float = 0.35, side_ratio: float = 0.6,
) -> List[List[float]]:
    """Background prompts ringing a head/tail line: past each end and off to
    both sides, far enough out to clear a deep-bodied object."""
    pts = np.asarray(line, dtype=np.float64)
    length = polyline_length(pts)
    chord = pts[-1] - pts[0]
    norm = float(np.linalg.norm(chord))
    if length <= 0 or norm <= 0:
        return []
    axis = chord / norm
    normal = np.array([-axis[1], axis[0]])
    ring = [pts[0] - axis * end_ratio * length, pts[-1] + axis * end_ratio * length]
    for fraction in (0.25, 0.5, 0.75):
        base = pts[0] + chord * fraction
        ring.append(base + normal * side_ratio * length)
        ring.append(base - normal * side_ratio * length)
    width, height = image_size
    return [
        [float(p[0]), float(p[1])] for p in ring
        if 0 <= p[0] < width and 0 <= p[1] < height
    ]


def clip_mask_to_line(
    mask: np.ndarray, offset: Tuple[float, float], line,
    half_width_ratio: float = 0.3,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Keep only the part of a cropped mask (top-left at offset, image
    coordinates) within half_width_ratio * line length of the line. Returns
    the largest surviving piece, re-cropped, with its new offset, or None
    when nothing is left."""
    length = polyline_length(line)
    if length <= 0 or mask.size == 0:
        return None
    binary = (mask[:, :, 0] if mask.ndim == 3 else mask) > 0
    local = np.round(np.asarray(line, dtype=np.float64) - np.asarray(offset)).astype(np.int32)
    region = np.zeros(binary.shape, dtype=np.uint8)
    thickness = max(1, int(round(2 * half_width_ratio * length)))
    image_kernels.draw_polyline(region, local.reshape(-1, 2).astype(np.float64),
                                1, False, thickness)
    clipped = (binary & (region > 0)).astype(np.uint8)
    count, labels = image_kernels.label_components(clipped, 8)
    if count < 2:
        return None
    # The band can cut a sprawling mask into scraps; only the largest is the
    # object. `cv2.connectedComponentsWithStats` reported the areas; counting
    # the labels gives the same thing, and label 0 is the background.
    areas = np.bincount(labels.ravel(), minlength=count)
    clipped = labels == 1 + int(np.argmax(areas[1:]))
    ys, xs = np.where(clipped)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    new_offset = (int(round(offset[0])) + int(x0), int(round(offset[1])) + int(y0))
    return clipped[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), new_offset


def mask_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """Connected components (8-connectivity) of a binary mask: (count, labels)."""
    count, labels = image_kernels.label_components(
        (mask > 0).astype(np.uint8), 8)
    return count - 1, labels


def point_in_mask(mask: np.ndarray, point) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def component_at(mask: np.ndarray, point) -> Optional[np.ndarray]:
    """The connected component of the mask holding the point, or None."""
    if not point_in_mask(mask, point):
        return None
    _, labels = mask_components(mask)
    return labels == labels[int(round(point[1])), int(round(point[0]))]


def disk_mask(shape: Tuple[int, int], point, radius: float) -> np.ndarray:
    ys, xs = np.ogrid[:shape[0], :shape[1]]
    return (xs - point[0]) ** 2 + (ys - point[1]) ** 2 <= radius ** 2


def simplify_polygon_within_error(
    polygon: List[List[float]],
    max_points: int = 25,
    max_points_limit: int = 100,
    max_error: float = 0.05,
    adaptive: bool = False,
) -> List[List[float]]:
    """
    Simplify to max_points, doubling that budget (up to max_points_limit)
    while the simplified ring's area differs from the original's by more than
    max_error of the original area.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    simplify = adaptive_simplify_polygon if adaptive else simplify_polygon_to_max_points
    if len(polygon) <= max_points:
        return polygon
    try:
        original = ShapelyPolygon(polygon).buffer(0)
    except Exception:
        return simplify(polygon, max_points)
    budget = max_points
    while True:
        result = simplify(polygon, budget)
        if budget >= max_points_limit or original.is_empty or original.area <= 0:
            return result
        try:
            error = original.symmetric_difference(ShapelyPolygon(result).buffer(0)).area
        except Exception:
            return result
        if error <= max_error * original.area:
            return result
        budget = min(max_points_limit, budget * 2)


class PromptInstances:
    """
    Creation buffer for point-click segmentation: one mask instance per object,
    built from prompts taken in click order.

    A positive click inside an instance refines it. One outside every instance
    joins the nearest instance whose joint prediction comes back as a single
    region holding all of its clicks, and otherwise starts a new instance from
    a prediction of its own, so every positive lies inside some instance and
    the output is simply as many polygons as that takes. Negative clicks apply
    to every instance; one the model still covers is carved out as a small disk.

    `predict(positives, negatives)` returns a full-frame binary mask or None.
    """

    def __init__(self, shape, predict, min_radius: int = 4, max_join_attempts: int = 3,
                 max_join_growth: float = 1.5):
        self.shape = tuple(shape)
        self.predict = predict
        self.radius = max(min_radius, int(round(0.01 * float(np.hypot(*self.shape)))))
        self.max_join_attempts = max_join_attempts
        self.max_join_growth = max_join_growth
        self.instances: List[dict] = []
        self.negatives: List = []
        self.prompts: List = []

    def sync(self, prompts) -> None:
        """Bring the buffer to `prompts` ([point, label] in click order),
        replaying from scratch unless they extend what is already applied."""
        prompts = [([float(p[0]), float(p[1])], int(label)) for p, label in prompts]
        if prompts[:len(self.prompts)] != self.prompts:
            self.instances, self.negatives, self.prompts = [], [], []
        for point, label in prompts[len(self.prompts):]:
            if label == 1:
                self._add_positive(point)
            else:
                self._add_negative(point)
            self.prompts.append((point, label))

    def mask(self) -> np.ndarray:
        total = np.zeros(self.shape, dtype=bool)
        for instance in self.instances:
            total |= instance["mask"]
        return total

    def _region(self, positives, connected: bool = False) -> Optional[np.ndarray]:
        """The predicted components holding the positives; None when any is
        missed, or (connected) when they do not share one component."""
        predicted = self.predict(positives, self.negatives)
        if predicted is None:
            return None
        predicted = np.asarray(predicted) > 0
        _, labels = mask_components(predicted)
        found = set()
        for point in positives:
            if not point_in_mask(predicted, point):
                return None
            found.add(labels[int(round(point[1])), int(round(point[0]))])
        if connected and len(found) > 1:
            return None
        return np.isin(labels, list(found))

    def _enforce(self, instance) -> None:
        for negative in self.negatives:
            if point_in_mask(instance["mask"], negative):
                instance["mask"] = instance["mask"] & ~disk_mask(self.shape, negative, self.radius)

    @staticmethod
    def _box_area(mask) -> float:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0.0
        return float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))

    def _add_positive(self, point) -> None:
        owner = next((i for i in self.instances if point_in_mask(i["mask"], point)), None)
        if owner is not None:
            refined = self._region(owner["positives"] + [point])
            owner["positives"].append(point)
            if refined is not None:
                owner["mask"] = refined
            self._enforce(owner)
            return

        alone = self._region([point])
        if alone is None:
            return

        def distance(instance):
            return min(np.hypot(p[0] - point[0], p[1] - point[1]) for p in instance["positives"])

        for instance in sorted(self.instances, key=distance)[:self.max_join_attempts]:
            joint = self._region(instance["positives"] + [point], connected=True)
            if joint is None:
                continue
            # A joint mask far larger than its parts swallowed background.
            if self._box_area(joint) > self.max_join_growth * self._box_area(instance["mask"] | alone):
                continue
            instance["positives"].append(point)
            instance["mask"] = joint
            self._enforce(instance)
            return

        instance = {"positives": [point], "mask": alone}
        self._enforce(instance)
        self.instances.append(instance)

    def _add_negative(self, point) -> None:
        self.negatives.append(point)
        for instance in self.instances:
            again = self._region(instance["positives"])
            if again is not None:
                instance["mask"] = again
            self._enforce(instance)
