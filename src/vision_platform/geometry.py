"""Small geometry helpers for zone logic. Polygons use normalized (0..1) coordinates."""

from __future__ import annotations

from vision_platform.types import BBox

Point = tuple[float, float]
Polygon = list[Point]


def foot_point(bbox: BBox) -> Point:
    """Bottom-center of a bbox: where a standing/walking person touches the floor."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def center_point(bbox: BBox) -> Point:
    """Bbox center: the zone anchor for seated/occluded tracks whose feet are hidden."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon; boundary counts as inside."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    # Boundary check: collinear and within segment bounds.
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-12 and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 \
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_at_y = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_at_y:
                inside = not inside
        j = i
    return inside
