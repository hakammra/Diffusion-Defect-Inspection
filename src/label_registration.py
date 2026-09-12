"""Register the fixed LabelInspect target using its four black fiducials."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


REFERENCE_SIZE = (1063, 650)
REFERENCE_MARKERS = np.asarray(
    [(49.0, 49.0), (1014.0, 49.0), (1014.0, 601.0), (49.0, 601.0)],
    dtype=np.float64,
)
DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "printed_labels" / "reference" / "label_master.png"
# Derived before opening the real test set: median minus five MADs across the 50
# registered training/validation normals (0.3649169, rounded down).
FOUR_MARKER_QUALITY_FLOOR = 0.36
# A content defect can lower reference agreement even when all four fiducials are
# correct.  Replace a four-marker warp only when the fallback is materially better.
FALLBACK_QUALITY_MARGIN = 0.08


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    width: int
    height: int
    area: int
    fill: float


def _components(mask: np.ndarray) -> list[Candidate]:
    """Return compact dark connected components from a small binary image."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    candidates: list[Candidate] = []
    for start_y, start_x in zip(*np.nonzero(mask & ~seen)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[next_y, next_x] and not seen[next_y, next_x]:
                        seen[next_y, next_x] = True
                        stack.append((next_y, next_x))
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        box_width, box_height = right - left + 1, bottom - top + 1
        area = len(xs)
        fill = area / (box_width * box_height)
        if (
            3 <= box_width <= 35
            and 3 <= box_height <= 35
            and 0.55 <= box_width / box_height <= 1.8
            and fill >= 0.42
            and area >= 7
        ):
            candidates.append(
                Candidate(
                    x=(left + right) / 2,
                    y=(top + bottom) / 2,
                    width=box_width,
                    height=box_height,
                    area=area,
                    fill=fill,
                )
            )
    return candidates


def _order_quad(points: np.ndarray) -> np.ndarray | None:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # Starting at top-left produces TL, TR, BR, BL for ordinary photographs.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    cross = []
    for index in range(4):
        a = ordered[(index + 1) % 4] - ordered[index]
        b = ordered[(index + 2) % 4] - ordered[(index + 1) % 4]
        cross.append(float(a[0] * b[1] - a[1] * b[0]))
    if not (all(value > 0 for value in cross) or all(value < 0 for value in cross)):
        return None
    return ordered


def _quad_score(quad: np.ndarray, sizes: np.ndarray, image_shape: tuple[int, int]) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    if min(top, right, bottom, left) < 27:
        return math.inf
    width = (top + bottom) / 2
    height = (left + right) / 2
    ratio = width / height
    if not 1.25 <= ratio <= 2.25:
        return math.inf
    edge_penalty = abs(top - bottom) / width + abs(left - right) / height
    ratio_penalty = abs(math.log(ratio / 1.748))
    diagonal_a = np.linalg.norm(quad[2] - quad[0])
    diagonal_b = np.linalg.norm(quad[3] - quad[1])
    diagonal_penalty = abs(diagonal_a - diagonal_b) / max(diagonal_a, diagonal_b)
    angle_penalty = 0.0
    for index in range(4):
        first = quad[(index - 1) % 4] - quad[index]
        second = quad[(index + 1) % 4] - quad[index]
        cosine = abs(float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))))
        angle_penalty += cosine / 4
    size_penalty = float(np.std(sizes) / max(np.mean(sizes), 1e-6))
    image_height, image_width = image_shape
    coverage = (width * height) / (image_width * image_height)
    if coverage < 0.025:
        return math.inf
    # Prefer a large, rectangular configuration of four similarly sized squares.
    return 2.0 * ratio_penalty + edge_penalty + diagonal_penalty + angle_penalty + size_penalty - 0.2 * coverage


def _marker_candidates(image: Image.Image, max_dimension: int = 500) -> tuple[list[Candidate], float, tuple[int, int]]:
    image = ImageOps.exif_transpose(image).convert("L")
    scale = min(1.0, max_dimension / max(image.size))
    small = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BILINEAR,
    )
    gray = np.asarray(small)
    # The printed fiducials are close to black even across the capture conditions.
    candidates = _components(gray < 65)
    candidates = sorted(candidates, key=lambda item: item.area, reverse=True)[:28]
    return candidates, scale, gray.shape


def locate_markers(image: Image.Image, max_dimension: int = 500) -> np.ndarray:
    candidates, scale, gray_shape = _marker_candidates(image, max_dimension=max_dimension)
    if len(candidates) < 4:
        raise ValueError(f"Only {len(candidates)} square marker candidates found")

    best_score = math.inf
    best_quad: np.ndarray | None = None
    for chosen in itertools.combinations(candidates, 4):
        points = np.asarray([(item.x, item.y) for item in chosen], dtype=np.float64)
        quad = _order_quad(points)
        if quad is None:
            continue
        # Reorder component sizes in the same way by matching point coordinates.
        sizes = []
        for point in quad:
            item = min(chosen, key=lambda value: (value.x - point[0]) ** 2 + (value.y - point[1]) ** 2)
            sizes.append(math.sqrt(item.width * item.height))
        score = _quad_score(quad, np.asarray(sizes), gray_shape)
        if score < best_score:
            best_score = score
            best_quad = quad
    if best_quad is None or best_score > 1.7:
        raise ValueError(f"No reliable four-marker rectangle found (best score {best_score:.3f})")
    return best_quad / scale


def _perspective_coefficients(destination: np.ndarray, source: np.ndarray) -> tuple[float, ...]:
    """Solve Pillow's output-to-input perspective mapping coefficients."""
    matrix = []
    values = []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])
    return tuple(np.linalg.solve(np.asarray(matrix), np.asarray(values)).tolist())


def _warp(image: Image.Image, markers: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    destination = REFERENCE_MARKERS.copy()
    destination[:, 0] *= output_size[0] / REFERENCE_SIZE[0]
    destination[:, 1] *= output_size[1] / REFERENCE_SIZE[1]
    coefficients = _perspective_coefficients(destination, markers)
    return image.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def registration_similarity(registered: Image.Image, reference_path: str | Path = DEFAULT_REFERENCE_PATH) -> float:
    """Measure fixed-layout agreement for registration quality control.

    This is not an anomaly score. It is used only to reject or recover implausible
    geometric warps before a sample reaches an anomaly model.
    """
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Registration reference not found: {reference_path}")
    with Image.open(reference_path) as source:
        reference = source.convert("L").resize(registered.size, Image.Resampling.BILINEAR)
    reference_dark = np.asarray(reference) < 120
    array = np.asarray(registered.convert("L"))
    threshold = min(145.0, float(np.quantile(array, 0.18)))
    observed_dark = array < threshold
    roi = np.zeros(reference_dark.shape, dtype=bool)
    border = max(2, round(min(registered.size) * 0.012))
    roi[border:-border, border:-border] = True
    intersection = int(np.sum(reference_dark & observed_dark & roi))
    denominator = int(np.sum(reference_dark & roi) + np.sum(observed_dark & roi))
    return 2 * intersection / denominator if denominator else 0.0


def _three_marker_alignment(
    image: Image.Image,
    output_size: tuple[int, int],
    reference_path: str | Path,
    max_dimension: int = 500,
) -> tuple[Image.Image, np.ndarray, float]:
    """Recover a label with one damaged fiducial using category-blind geometry."""
    candidates, scale, gray_shape = _marker_candidates(image, max_dimension=max_dimension)
    candidates = candidates[:18]
    if len(candidates) < 3:
        raise ValueError(f"Only {len(candidates)} square marker candidates found")

    scored_geometry: list[tuple[float, np.ndarray]] = []
    source_height, source_width = image.height, image.width
    for chosen in itertools.combinations(candidates, 3):
        source_unordered = np.asarray([(item.x / scale, item.y / scale) for item in chosen], dtype=np.float64)
        marker_sizes = np.asarray([math.sqrt(item.width * item.height) / scale for item in chosen])
        # True fiducials have the same printed dimensions.  This rejects small,
        # square-looking fragments from letters or barcode bars that can form a
        # geometrically plausible but severely cropped label.
        if float(np.max(marker_sizes) / max(np.min(marker_sizes), 1e-6)) > 1.65:
            continue
        size_penalty = float(np.std(marker_sizes) / max(np.mean(marker_sizes), 1e-6))
        for missing_index in range(4):
            present_indices = [index for index in range(4) if index != missing_index]
            destination = REFERENCE_MARKERS[present_indices]
            for permutation in itertools.permutations(range(3)):
                source = source_unordered[list(permutation)]
                design = np.column_stack([destination, np.ones(3)])
                affine_x = np.linalg.solve(design, source[:, 0])
                affine_y = np.linalg.solve(design, source[:, 1])
                # The capture protocol permits rotation and perspective, but never
                # a reflection.  Reject mirrored corner assignments before image
                # similarity scoring; a reflected label can otherwise overlap the
                # reference enough to beat the correct three-marker hypothesis.
                determinant = affine_x[0] * affine_y[1] - affine_x[1] * affine_y[0]
                if determinant <= 0:
                    continue
                full_design = np.column_stack([REFERENCE_MARKERS, np.ones(4)])
                predicted = np.column_stack([full_design @ affine_x, full_design @ affine_y])
                if (
                    np.any(predicted[:, 0] < -0.05 * source_width)
                    or np.any(predicted[:, 0] > 1.05 * source_width)
                    or np.any(predicted[:, 1] < -0.05 * source_height)
                    or np.any(predicted[:, 1] > 1.05 * source_height)
                ):
                    continue
                sizes = np.insert(marker_sizes, missing_index, float(np.median(marker_sizes)))
                geometry = _quad_score(predicted, sizes, (source_height, source_width)) + size_penalty
                if math.isfinite(geometry) and geometry < 1.8:
                    scored_geometry.append((geometry, predicted))
    if not scored_geometry:
        raise ValueError("No plausible three-marker alignment found")

    # Expensive image agreement is evaluated only on the strongest geometric candidates.
    best: tuple[float, float, np.ndarray] | None = None
    score_size = (266, 163)
    for geometry, markers in sorted(scored_geometry, key=lambda item: item[0])[:36]:
        probe = _warp(image, markers, score_size)
        quality = registration_similarity(probe, reference_path)
        ranking = (quality, -geometry)
        if best is None or ranking > (best[0], best[1]):
            best = (quality, -geometry, markers)
    assert best is not None
    quality, _, markers = best
    if quality < 0.28:
        raise ValueError(f"Three-marker alignment quality remained too low ({quality:.3f})")
    return _warp(image, markers, output_size), markers, quality


def register_label_detailed(
    path: str | Path,
    output_size: tuple[int, int] = REFERENCE_SIZE,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> tuple[Image.Image, np.ndarray, dict]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    four_error: Exception | None = None
    four_result: tuple[Image.Image, np.ndarray, float] | None = None
    try:
        markers = locate_markers(image)
        registered = _warp(image, markers, output_size)
        quality = registration_similarity(registered, reference_path)
        four_result = (registered, markers, quality)
        if quality >= FOUR_MARKER_QUALITY_FLOOR:
            return registered, markers, {"mode": "four_marker", "quality": quality}
    except Exception as error:
        four_error = error

    try:
        registered, markers, quality = _three_marker_alignment(image, output_size, reference_path)
        if four_result is not None and quality < four_result[2] + FALLBACK_QUALITY_MARGIN:
            registered, markers, quality = four_result
            return registered, markers, {"mode": "four_marker_low_quality", "quality": quality}
        return registered, markers, {"mode": "three_marker_fallback", "quality": quality}
    except Exception as fallback_error:
        if four_result is not None:
            registered, markers, quality = four_result
            return registered, markers, {
                "mode": "four_marker_low_quality",
                "quality": quality,
                "fallback_error": str(fallback_error),
            }
        raise ValueError(
            f"Four-marker registration failed ({four_error}); three-marker fallback failed ({fallback_error})"
        ) from fallback_error


def register_label(path: str | Path, output_size: tuple[int, int] = REFERENCE_SIZE) -> tuple[Image.Image, np.ndarray]:
    registered, markers, _ = register_label_detailed(path, output_size=output_size)
    return registered, markers
