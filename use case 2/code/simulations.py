#!/usr/bin/env python3
"""
Boundary-case simulator for the two-anchor disambiguation method.

What this script does:
1) Simulates two anchors and a moving phone pose (position + heading).
2) Computes ground-truth relative angles to both anchors.
3) Applies different measurement behaviors (ideal, wrap, noisy wrap, etc.).
4) Runs the same two-anchor disambiguation rule used in the paper.
5) Produces an exhaustive combination table over GT/measured angle bins:
   (gt_bin_a1, gt_bin_a2, meas_bin_a1, meas_bin_a2)
   with rates of:
   - correct intersection chosen
   - wrong intersection chosen
   - no intersection (range circles do not intersect)

Outputs:
- CSV with exhaustive combinations and rates.
- Optional terminal summary of highest-failure combinations.

Usage example:
  python simulate_two_anchor_boundary_cases.py \
      --samples 120000 \
      --scenario wrap_noise \
      --out two_anchor_exhaustive_wrap_noise.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # optional dependency guard
    wilcoxon = None

try:
    from scipy.stats import binomtest
except Exception:  # optional dependency guard
    binomtest = None


# Angle bins used for exhaustive case tables.
# FR: front-right, FL: front-left, BR: back-right, BL: back-left.
ANGLE_BINS = ["FR", "FL", "BR", "BL"]


@dataclass
class SimResult:
    gt_bin_a1: str
    gt_bin_a2: str
    meas_bin_a1: str
    meas_bin_a2: str
    outcome: str  # "correct", "wrong", "no_intersection", "degenerate"
    chosen_pos_error_m: float = math.nan
    random_pos_error_m: float = math.nan
    random_correct: Optional[bool] = None
    kinematic_correct: Optional[bool] = None
    kinematic_inter_sep_m: float = math.nan
    kinematic_crossed_baseline: Optional[bool] = None
    kinematic_dist_to_correct_m: float = math.nan
    kinematic_dist_to_wrong_m: float = math.nan
    kinematic_prev_pos_error_m: float = math.nan
    kinematic_step_size_m: float = math.nan
    kinematic_chosen_pos_error_m: float = math.nan
    kinematic_random_correct: Optional[bool] = None
    no_int_random_err_m: float = math.nan


def _get_no_int_candidate_points(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float,
) -> Optional[List[Tuple[float, float]]]:
    """Return the two closest points on each circle along the inter-centre line."""
    x1, y1 = c1
    x2, y2 = c2
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    ux = dx / d
    uy = dy / d
    if d > r1 + r2:
        p1 = (x1 + r1 * ux, y1 + r1 * uy)
        p2 = (x2 - r2 * ux, y2 - r2 * uy)
    elif d < abs(r1 - r2):
        if r1 >= r2:
            p1 = (x1 + r1 * ux, y1 + r1 * uy)
            p2 = (x2 + r2 * ux, y2 + r2 * uy)
        else:
            p1 = (x1 - r1 * ux, y1 - r1 * uy)
            p2 = (x2 - r2 * ux, y2 - r2 * uy)
    else:
        p1 = (x1 + r1 * ux, y1 + r1 * uy)
        p2 = (x2 - r2 * ux, y2 - r2 * uy)
    return [p1, p2]


def closest_points_fallback(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float,
    range_variance: float,
) -> Tuple[Tuple[float, float], bool]:
    """Fallback when circles do not intersect — weighted midpoint of closest points."""
    x1, y1 = c1
    x2, y2 = c2
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    eps = 1e-9

    if d < eps:
        if abs(r1 - r2) < eps:
            return (0.0, 0.0), False
        rmin = min(r1, r2)
        return (x1 + rmin, y1), True

    ux = dx / d
    uy = dy / d

    if d > r1 + r2:
        p1 = (x1 + r1 * ux, y1 + r1 * uy)
        p2 = (x2 - r2 * ux, y2 - r2 * uy)
    elif d < abs(r1 - r2):
        if r1 >= r2:
            p1 = (x1 + r1 * ux, y1 + r1 * uy)
            p2 = (x2 + r2 * ux, y2 + r2 * uy)
        else:
            p1 = (x1 - r1 * ux, y1 - r1 * uy)
            p2 = (x2 - r2 * ux, y2 - r2 * uy)
    else:
        p1 = (x1 + r1 * ux, y1 + r1 * uy)
        p2 = (x2 - r2 * ux, y2 - r2 * uy)

    w1 = 1.0 / max(range_variance, eps)
    w2 = w1
    w_total = w1 + w2
    fx = (w1 * p1[0] + w2 * p2[0]) / w_total
    fy = (w1 * p1[1] + w2 * p2[1]) / w_total
    return (fx, fy), True


def wrap_to_180(deg: float) -> float:
    """Wrap any angle in degrees to [-180, 180)."""
    out = (deg + 180.0) % 360.0 - 180.0
    return out


def is_front_hemisphere(deg: float) -> bool:
    """True if angle is in the front hemisphere, i.e., |angle| <= 90 deg."""
    return abs(wrap_to_180(deg)) <= 90.0


def mirror_across_front_back_boundary(deg: float) -> float:
    """
    Mirror angle across the front/back boundary while preserving left/right sign.

    This toggles front <-> behind without changing side:
    - +80 -> +100
    - -80 -> -100
    - +120 -> +60
    - -120 -> -60
    """
    d = wrap_to_180(deg)
    return wrap_to_180(math.copysign(180.0 - abs(d), d))


def circular_abs_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angular difference in degrees, always in [0, 180]."""
    return abs(wrap_to_180(a_deg - b_deg))


def angle_bin(deg: float) -> str:
    """Map an angle in [-180, 180) to one of four directional bins."""
    d = wrap_to_180(deg)
    if 0.0 <= d < 90.0:
        return "FR"
    if -90.0 <= d < 0.0:
        return "FL"
    if 90.0 <= d < 180.0:
        return "BR"
    return "BL"  # [-180, -90)


def wrap_to_fov_pm90(deg: float) -> float:
    """
    Simulate the UWB front-back (phase) ambiguity model.

    Front region (|angle| <= 90): angle unchanged.
    Behind region (|angle| > 90): angle is reflected across the ±90 boundary
    so that true + wrapped = -180 (negative) or +180 (positive).

    This preserves the left/right sign while placing the measured angle
    in the front hemisphere (|result| <= 90).

    Examples:
    - -120 ->  -60   ( -120 + -60 = -180 )
    - +120 ->  +60   (  120 +  60 =  180 )
    - -170 ->  -10   ( -170 + -10 = -180 )
    - +100 ->  +80   (  100 +  80 =  180 )
    -  +40 ->  +40   (no change, already in front)
    """
    d = wrap_to_180(deg)
    ad = abs(d)
    if ad <= 90.0:
        return d
    return math.copysign(180.0 - ad, d)


def relative_angle_deg(phone_x: float, phone_y: float, heading_deg: float, anchor: Tuple[float, float]) -> float:
    """Signed relative angle from phone heading to anchor bearing in degrees."""
    ax, ay = anchor
    bearing = math.degrees(math.atan2(ay - phone_y, ax - phone_x))
    return wrap_to_180(bearing - heading_deg)


def circle_intersections(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float,
    eps: float = 1e-9,
) -> Optional[List[Tuple[float, float]]]:
    """Return intersections of two circles (0, 1, or 2 points)."""
    x0, y0 = c1
    x1, y1 = c2
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)

    # No intersection or concentric.
    if d < eps:
        return None
    if d > r1 + r2 + eps:
        return None
    if d < abs(r1 - r2) - eps:
        return None

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < -eps:
        return None
    h = math.sqrt(max(0.0, h_sq))

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)

    if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < eps:
        return [p1]
    return [p1, p2]


def circle_intersections_dilateration_style(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    Same closed-form circle intersection routine used in code/dataset.py
    (get_two_circles_intersections).
    """
    x1, y1 = c1
    x2, y2 = c2
    centerdx = x1 - x2
    centerdy = y1 - y2
    R = math.sqrt(centerdx * centerdx + centerdy * centerdy)

    if not (abs(r1 - r2) <= R and (R <= r1 + r2)):
        return None

    if R == 0.0:
        return None

    R2 = R * R
    R4 = R2 * R2
    a = (r1 * r1 - r2 * r2) / (2.0 * R2)
    r2r2 = (r1 * r1 - r2 * r2)
    c_expr = 2.0 * (r1 * r1 + r2 * r2) / R2 - (r2r2 * r2r2) / R4 - 1.0
    if c_expr < 0.0:
        return None
    c = math.sqrt(c_expr)

    fx = (x1 + x2) / 2.0 + a * (x2 - x1)
    gx = c * (y2 - y1) / 2.0
    ix1 = fx + gx
    ix2 = fx - gx

    fy = (y1 + y2) / 2.0 + a * (y2 - y1)
    gy = c * (x1 - x2) / 2.0
    iy1 = fy + gy
    iy2 = fy - gy

    return [(ix1, iy1), (ix2, iy2)]


def rotate_vec_ccw(vx: float, vy: float, deg: float) -> Tuple[float, float]:
    """Rotate vector (vx, vy) anti-clockwise by deg."""
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)
    return c * vx - s * vy, s * vx + c * vy


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float], eps: float = 1e-9) -> float:
    """Angle between vectors in [0, 180] degrees."""
    x1, y1 = v1
    x2, y2 = v2
    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)
    if n1 < eps or n2 < eps:
        return 180.0
    cosv = (x1 * x2 + y1 * y2) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def choose_intersection_by_orientation_consistency(
    intersections: List[Tuple[float, float]],
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    meas_rel_a1_deg: float,
    meas_rel_a2_deg: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Selection rule requested by user:
    choose the candidate whose two orientation estimates have the smallest
    angular difference.

    For each candidate intersection i:
    - build vectors i->A1 and i->A2
    - rotate by -ma1 and -ma2 respectively
    - compute angle(ori1, ori2)
    - pick candidate with the smaller angle
    """
    if not intersections:
        return None, None

    best_idx = None
    best_error = None

    for i, p in enumerate(intersections):
        px, py = p
        # Candidate-to-anchor vectors.
        v1 = (a1[0] - px, a1[1] - py)
        v2 = (a2[0] - px, a2[1] - py)

        # Same sign convention as dataset.dilateration.
        phi1 = rotate_vec_ccw(v1[0], v1[1], -meas_rel_a1_deg)
        phi2 = rotate_vec_ccw(v2[0], v2[1], -meas_rel_a2_deg)

        err = angle_between(phi1, phi2)
        if best_error is None or err < best_error:
            best_error = err
            best_idx = i

    return best_idx, best_error


def apply_measurement_model(gt_angle_deg: float, scenario: str, sigma_deg: float, mu_deg: float = 0.0) -> float:
    """Generate measured relative angle according to selected scenario."""
    g = wrap_to_180(gt_angle_deg)

    if scenario == "ideal":
        m = g
    elif scenario == "wrap":
        m = wrap_to_fov_pm90(g)
    elif scenario == "wrap_noise":
        # Apply noise first. Emulate front-back ambiguity only when the
        # anchor is truly behind and the noisy angle is also behind.
        noisy = wrap_to_180(g + random.gauss(mu_deg, sigma_deg))
        if abs(g) > 90.0 and abs(noisy) > 90.0:
            m = wrap_to_fov_pm90(noisy)
        else:
            m = noisy
    elif scenario == "ideal_noise":
        m = g + random.gauss(mu_deg, sigma_deg)
    elif scenario == "flip_wrap":
        # Severe ambiguity model: sign inversion + FOV wrapping.
        m = wrap_to_fov_pm90(-g)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return wrap_to_180(m)


def apply_measurement_model_no_fb_resolved(
    gt_angle_deg: float,
    sigma_deg: float,
    mu_deg: float = 0.0,
    enforce_hemisphere_consistency: bool = True,
) -> float:
    """
    No-front-back-ambiguity model.

    Noise is injected first. Optionally, if noise alone crosses the front/back
    boundary, mirror the noisy angle back to the GT hemisphere to model a
    front-back-resolved sensor.
    """
    g = wrap_to_180(gt_angle_deg)
    m = wrap_to_180(g + random.gauss(mu_deg, sigma_deg))

    if enforce_hemisphere_consistency and (is_front_hemisphere(m) != is_front_hemisphere(g)):
        m = mirror_across_front_back_boundary(m)

    return wrap_to_180(m)


def run_simulation(
    samples: int,
    scenario: str,
    anchor_mode: str,
    fixed_anchor_distance: float,
    min_anchor_separation_m: float,
    range_sigma_m: float,
    range_mu_m: float,
    angle_sigma_deg: float,
    angle_mu_deg: float,
    seed: int,
    min_distance_m: float,
    max_distance_m: float,
    exclude_front_to_behind_misreports: bool,
    enforce_no_fb_hemisphere_consistency: bool = True,
    kinematic_fallback: bool = False,
    kinematic_step_mu: float = 0.7,
    kinematic_prev_pos_sigma: float = 0.5,
) -> List[SimResult]:
    random.seed(seed)

    results: List[SimResult] = []

    # Sample phone poses while enforcing distance constraints to both anchors.
    attempts = 0
    max_attempts = max(10 * samples, 1)
    while len(results) < samples and attempts < max_attempts:
        attempts += 1

        # Anchor placement mode.
        if anchor_mode == "fixed":
            a1 = (-fixed_anchor_distance / 2.0, 0.0)
            a2 = (fixed_anchor_distance / 2.0, 0.0)
        elif anchor_mode == "random":
            # Randomize anchor coordinates per case to avoid fixed-geometry bias.
            # Keep anchors within the same spatial envelope used for targets.
            placed = False
            for _a_attempt in range(2000):
                a1 = (
                    random.uniform(-max_distance_m, max_distance_m),
                    random.uniform(-max_distance_m, max_distance_m),
                )
                a2 = (
                    random.uniform(-max_distance_m, max_distance_m),
                    random.uniform(-max_distance_m, max_distance_m),
                )
                if math.hypot(a1[0] - a2[0], a1[1] - a2[1]) >= min_anchor_separation_m:
                    placed = True
                    break
            if not placed:
                continue
        else:
            raise ValueError(f"Unknown anchor mode: {anchor_mode}")

        accepted = False
        px = py = 0.0
        r1 = r2 = 0.0
        for _attempt in range(2000):
            px = random.uniform(-max_distance_m, max_distance_m)
            py = random.uniform(-max_distance_m, max_distance_m)
            r1 = math.hypot(a1[0] - px, a1[1] - py)
            r2 = math.hypot(a2[0] - px, a2[1] - py)
            if min_distance_m <= r1 <= max_distance_m and min_distance_m <= r2 <= max_distance_m:
                accepted = True
                break
        if not accepted:
            # In practice, this should not happen with the current bounds.
            continue

        heading = random.uniform(-180.0, 180.0)

        # True ranges and true relative angles.
        gt1 = relative_angle_deg(px, py, heading, a1)
        gt2 = relative_angle_deg(px, py, heading, a2)

        # Measured ranges.
        mr1 = max(1e-3, r1 + random.gauss(range_mu_m, range_sigma_m))
        mr2 = max(1e-3, r2 + random.gauss(range_mu_m, range_sigma_m))

        # Measured angles.
        if scenario == "ideal_noise":
            m1 = apply_measurement_model_no_fb_resolved(
                gt1,
                angle_sigma_deg,
                angle_mu_deg,
                enforce_hemisphere_consistency=enforce_no_fb_hemisphere_consistency,
            )
            m2 = apply_measurement_model_no_fb_resolved(
                gt2,
                angle_sigma_deg,
                angle_mu_deg,
                enforce_hemisphere_consistency=enforce_no_fb_hemisphere_consistency,
            )
        else:
            m1 = apply_measurement_model(gt1, scenario, angle_sigma_deg, angle_mu_deg)
            m2 = apply_measurement_model(gt2, scenario, angle_sigma_deg, angle_mu_deg)

        # Optional realism filter: if anchor is in front, UWB is unlikely to report it as behind.
        if exclude_front_to_behind_misreports:
            impossible_1 = (abs(gt1) < 90.0) and (abs(m1) > 90.0)
            impossible_2 = (abs(gt2) < 90.0) and (abs(m2) > 90.0)
            if impossible_1 or impossible_2:
                continue

        # Physical sanity: smallest GT-measured angular separation cannot exceed 180 deg.
        if circular_abs_diff_deg(gt1, m1) > 180.0 + 1e-9 or circular_abs_diff_deg(gt2, m2) > 180.0 + 1e-9:
            raise RuntimeError("Invalid angle generation: circular GT-measured difference exceeded 180 degrees")

        gt_bin_1 = angle_bin(gt1)
        gt_bin_2 = angle_bin(gt2)
        m_bin_1 = angle_bin(m1)
        m_bin_2 = angle_bin(m2)

        # Use the same geometric solver as dataset.dilateration.
        inter = circle_intersections_dilateration_style(a1, mr1, a2, mr2)
        range_variance = range_sigma_m * range_sigma_m
        if not inter:
            fb_pt, fb_ok = closest_points_fallback(a1, mr1, a2, mr2, range_variance)
            fb_err = math.hypot(fb_pt[0] - px, fb_pt[1] - py) if fb_ok else math.nan
            no_int_rand_err = math.nan
            if fb_ok:
                pts = _get_no_int_candidate_points(a1, mr1, a2, mr2)
                if pts:
                    rp = pts[random.randint(0, len(pts) - 1)]
                    no_int_rand_err = math.hypot(rp[0] - px, rp[1] - py)
            results.append(
                SimResult(gt_bin_1, gt_bin_2, m_bin_1, m_bin_2, "no_intersection",
                          fb_err, math.nan, None,
                          no_int_random_err_m=no_int_rand_err)
            )
            continue

        # If only one intersection, geometric ambiguity is absent.
        if len(inter) == 1:
            d_single = math.hypot(inter[0][0] - px, inter[0][1] - py)
            results.append(SimResult(gt_bin_1, gt_bin_2, m_bin_1, m_bin_2, "degenerate", d_single, math.nan, True))
            continue

        chosen_idx, _ = choose_intersection_by_orientation_consistency(inter, a1, a2, m1, m2)
        if chosen_idx is None:
            results.append(SimResult(gt_bin_1, gt_bin_2, m_bin_1, m_bin_2, "wrong", math.nan, math.nan, None))
            continue

        # Correct reference: the intersection closer to true phone position.
        d0 = math.hypot(inter[0][0] - px, inter[0][1] - py)
        d1 = math.hypot(inter[1][0] - px, inter[1][1] - py)
        correct_idx = 0 if d0 <= d1 else 1
        chosen_err = d0 if chosen_idx == 0 else d1
        random_idx = random.randint(0, 1)
        random_err = d0 if random_idx == 0 else d1
        random_correct = random_idx == correct_idx

        outcome = "correct" if chosen_idx == correct_idx else "wrong"

        k_correct = None
        k_inter_sep = math.nan
        k_crossed = None
        k_dist_correct = math.nan
        k_dist_wrong = math.nan
        k_prev_err = math.nan
        k_step = math.nan
        k_chosen_pos_err = math.nan
        k_rand_correct = None

        if kinematic_fallback and inter and len(inter) == 2:
            k_inter_sep = math.hypot(inter[0][0] - inter[1][0], inter[0][1] - inter[1][1])
            k_step = kinematic_step_mu * (0.8 + 0.4 * random.random())
            hdg_rad = math.radians(heading)
            prev_x = px - k_step * math.cos(hdg_rad)
            prev_y = py - k_step * math.sin(hdg_rad)
            kpx = prev_x + random.gauss(0.0, kinematic_prev_pos_sigma)
            kpy = prev_y + random.gauss(0.0, kinematic_prev_pos_sigma)
            k_prev_err = math.hypot(kpx - prev_x, kpy - prev_y)

            d_c0 = math.hypot(inter[0][0] - kpx, inter[0][1] - kpy)
            d_c1 = math.hypot(inter[1][0] - kpx, inter[1][1] - kpy)
            chosen_kin = 0 if d_c0 <= d_c1 else 1
            k_correct = chosen_kin == correct_idx
            k_chosen_pos_err = math.hypot(inter[chosen_kin][0] - px, inter[chosen_kin][1] - py)
            k_rand_correct = random.randint(0, 1) == correct_idx

            wrong_idx = 1 - correct_idx
            k_dist_correct = math.hypot(inter[correct_idx][0] - kpx, inter[correct_idx][1] - kpy)
            k_dist_wrong = math.hypot(inter[wrong_idx][0] - kpx, inter[wrong_idx][1] - kpy)

            bx, by = a2[0] - a1[0], a2[1] - a1[1]
            def _side(pt):
                return 1 if bx * (pt[1] - a1[1]) - by * (pt[0] - a1[0]) >= 0 else 0
            k_crossed = _side((px, py)) != _side((prev_x, prev_y))

        results.append(SimResult(
            gt_bin_1, gt_bin_2, m_bin_1, m_bin_2, outcome,
            chosen_err, random_err, random_correct,
            k_correct, k_inter_sep, k_crossed, k_dist_correct,
            k_dist_wrong, k_prev_err, k_step, k_chosen_pos_err, k_rand_correct,
        ))

    if len(results) < samples:
        raise RuntimeError(
            "Could not generate enough valid samples under current constraints. "
            "Try increasing max-distance, relaxing filters, or reducing requested samples."
        )

    return results


def compute_requested_success_rates(results: List[SimResult]) -> Dict[str, float]:
    """Compute requested success rates from simulation outcomes."""
    total = len(results)
    correct = sum(1 for r in results if r.outcome == "correct")
    no_inter = sum(1 for r in results if r.outcome == "no_intersection")

    # 1) All cases.
    all_cases = (correct / total) if total > 0 else float("nan")

    # 2) Only when intersections exist (exclude no_intersection).
    with_intersections_total = total - no_inter
    with_intersections_correct = sum(1 for r in results if r.outcome in ("correct", "degenerate"))
    intersections_exist = (
        with_intersections_correct / with_intersections_total if with_intersections_total > 0 else float("nan")
    )

    # 3) Only when both anchors are in front (|GT angle| < 90), i.e., bins FR/FL for both anchors.
    front_bins = {"FR", "FL"}
    front_subset = [r for r in results if r.gt_bin_a1 in front_bins and r.gt_bin_a2 in front_bins]
    front_total = len(front_subset)
    front_correct = sum(1 for r in front_subset if r.outcome == "correct")
    both_front = (front_correct / front_total) if front_total > 0 else float("nan")

    return {
        "all_cases": all_cases,
        "intersections_exist": intersections_exist,
        "both_anchors_front": both_front,
        "count_total": float(total),
        "count_with_intersections": float(with_intersections_total),
        "count_both_front": float(front_total),
    }


def compute_front_back_breakdown(results: List[SimResult]) -> Dict[str, Dict[str, float]]:
    """Break down success by GT front/behind anchor categories."""
    front_bins = {"FR", "FL"}
    behind_bins = {"BR", "BL"}

    def category_of(r: SimResult) -> str:
        a1_front = r.gt_bin_a1 in front_bins
        a2_front = r.gt_bin_a2 in front_bins
        a1_behind = r.gt_bin_a1 in behind_bins
        a2_behind = r.gt_bin_a2 in behind_bins

        if a1_front and a2_front:
            return "both_front"
        if a1_behind and a2_behind:
            return "both_behind"
        return "one_front_one_behind"

    buckets: Dict[str, List[SimResult]] = {
        "both_front": [],
        "one_front_one_behind": [],
        "both_behind": [],
    }

    for r in results:
        buckets[category_of(r)].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for key, vals in buckets.items():
        n = len(vals)
        c = sum(1 for r in vals if r.outcome == "correct")
        out[key] = {
            "success_rate": (c / n) if n > 0 else float("nan"),
            "count": float(n),
        }
    return out


def compute_position_error_stats(
    results: List[SimResult],
    include_outcomes: Tuple[str, ...] = ("correct", "degenerate"),
) -> Dict[str, float]:
    """Compute mean, std, 95% CI, and P90 positioning error for selected outcomes."""
    errs = [
        r.chosen_pos_error_m
        for r in results
        if r.outcome in include_outcomes and not math.isnan(r.chosen_pos_error_m)
    ]
    return _compute_error_stats_from_list(errs)


def compute_position_error_stats_all(
    results: List[SimResult],
) -> Dict[str, float]:
    """Compute positioning error stats over ALL trials (incl. wrong intersections and no-intersection fallback)."""
    errs = [
        r.chosen_pos_error_m
        for r in results
        if not math.isnan(r.chosen_pos_error_m)
    ]
    return _compute_error_stats_from_list(errs)


def _compute_error_stats_from_list(errs: List[float]) -> Dict[str, float]:
    """Shared stats computation for a list of errors."""
    if not errs:
        return {
            "count": 0.0, "mean": float("nan"), "std": float("nan"),
            "ci95_lo": float("nan"), "ci95_hi": float("nan"), "p90": float("nan"),
        }
    errs_s = sorted(errs)
    n = len(errs_s)
    mean_v = sum(errs_s) / n
    if n > 1:
        var = sum((x - mean_v) ** 2 for x in errs_s) / (n - 1)
        std_v = math.sqrt(var)
        ci_half = 1.959963984540054 * (std_v / math.sqrt(n))
        ci_lo = mean_v - ci_half
        ci_hi = mean_v + ci_half
    else:
        std_v = 0.0
        ci_lo = mean_v
        ci_hi = mean_v
    p = 0.90
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    p90_v = errs_s[lo] if lo == hi else errs_s[lo] * (1.0 - (pos - lo)) + errs_s[hi] * (pos - lo)
    return {
        "count": float(n),
        "mean": mean_v,
        "std": std_v,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p90": p90_v,
    }


def compute_wilcoxon_vs_random(results: List[SimResult]) -> Dict[str, float]:
    """Paired Wilcoxon test comparing heuristic vs random baseline on ambiguous cases."""
    pairs = [
        (r.chosen_pos_error_m, r.random_pos_error_m)
        for r in results
        if not math.isnan(r.chosen_pos_error_m) and not math.isnan(r.random_pos_error_m)
    ]

    if len(pairs) == 0:
        return {
            "count": 0.0,
            "heuristic_mean": float("nan"),
            "random_mean": float("nan"),
            "mean_diff_random_minus_heuristic": float("nan"),
            "w_stat": float("nan"),
            "p_value": float("nan"),
        }

    heuristic = np.array([p[0] for p in pairs], dtype=float)
    random_baseline = np.array([p[1] for p in pairs], dtype=float)
    diff = random_baseline - heuristic  # positive => heuristic better

    w_stat = float("nan")
    p_value = float("nan")
    if wilcoxon is not None:
        try:
            w_raw, p_raw = wilcoxon(diff, zero_method="pratt", alternative="greater", method="approx")
            w_stat = float(w_raw)
            p_value = float(p_raw)
        except Exception:
            pass

    return {
        "count": float(len(pairs)),
        "heuristic_mean": float(np.mean(heuristic)),
        "random_mean": float(np.mean(random_baseline)),
        "mean_diff_random_minus_heuristic": float(np.mean(diff)),
        "w_stat": w_stat,
        "p_value": p_value,
    }


def compute_mcnemar_vs_random(results: List[SimResult]) -> Dict[str, float]:
    """McNemar test on paired intersection-choice accuracy versus random baseline.

    The contingency table is built only from ambiguous two-intersection cases,
    where both method and random baseline can make a binary choice.
    """
    # a: method correct, baseline correct
    # b: method correct, baseline incorrect
    # c: method incorrect, baseline correct
    # d: method incorrect, baseline incorrect
    a = b = c = d = 0

    for r in results:
        if r.outcome not in ("correct", "wrong"):
            continue
        if r.random_correct is None:
            continue

        method_correct = r.outcome == "correct"
        baseline_correct = bool(r.random_correct)

        if method_correct and baseline_correct:
            a += 1
        elif method_correct and not baseline_correct:
            b += 1
        elif (not method_correct) and baseline_correct:
            c += 1
        else:
            d += 1

    n = a + b + c + d
    discordant = b + c
    method_acc = (a + b) / n if n > 0 else float("nan")
    baseline_acc = (a + c) / n if n > 0 else float("nan")

    # Continuity-corrected McNemar statistic.
    if discordant > 0:
        chi2_cc = ((abs(b - c) - 1.0) ** 2) / discordant
    else:
        chi2_cc = float("nan")

    # Exact one-sided p-value with H1: method is better than baseline (b > c).
    p_exact_one_sided = float("nan")
    if binomtest is not None and discordant > 0:
        try:
            p_exact_one_sided = float(binomtest(k=b, n=discordant, p=0.5, alternative="greater").pvalue)
        except Exception:
            pass

    return {
        "count": float(n),
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "discordant": float(discordant),
        "method_accuracy": method_acc,
        "baseline_accuracy": baseline_acc,
        "chi2_cc": chi2_cc,
        "p_exact_one_sided": p_exact_one_sided,
    }


def compute_kinematic_fallback_stats(results: List[SimResult]) -> Dict:
    valid = [r for r in results if r.kinematic_correct is not None]
    if not valid:
        return {}
    n = len(valid)
    n_correct = sum(1 for r in valid if r.kinematic_correct)
    n_cross = sum(1 for r in valid if r.kinematic_crossed_baseline)
    no_cross = [r for r in valid if not r.kinematic_crossed_baseline]
    n_nc = len(no_cross)
    c_nc = sum(1 for r in no_cross if r.kinematic_correct)
    cross_sub = [r for r in valid if r.kinematic_crossed_baseline]
    n_cr = len(cross_sub)
    c_cr = sum(1 for r in cross_sub if r.kinematic_correct)
    sep_bins = [(0, 1), (1, 3), (3, 10), (10, 100)]
    sep_stats = {}
    for lo, hi in sep_bins:
        sub = [r for r in valid if lo <= r.kinematic_inter_sep_m < hi]
        if sub:
            sep_stats[f"{lo}-{hi}m"] = {
                "n": len(sub),
                "pct": 100.0 * len(sub) / n,
                "acc": sum(1 for r in sub if r.kinematic_correct) / len(sub),
            }
    return {
        "n": n,
        "accuracy": n_correct / n,
        "accuracy_no_cross": c_nc / max(n_nc, 1),
        "accuracy_crossing": c_cr / max(n_cr, 1),
        "pct_crossing": 100.0 * n_cross / n,
        "avg_dist_correct_m": float(np.mean([r.kinematic_dist_to_correct_m for r in valid])),
        "avg_dist_wrong_m": float(np.mean([r.kinematic_dist_to_wrong_m for r in valid])),
        "avg_inter_sep_m": float(np.mean([r.kinematic_inter_sep_m for r in valid])),
        "avg_prev_error_m": float(np.mean([r.kinematic_prev_pos_error_m for r in valid])),
        "by_intersection_separation": sep_stats,
    }


def compute_no_int_wilcoxon_vs_random(results: List[SimResult]) -> Dict[str, float]:
    """Paired Wilcoxon test comparing no-int fallback vs random closest-point pick."""
    pairs = [
        (r.chosen_pos_error_m, r.no_int_random_err_m)
        for r in results
        if r.outcome == "no_intersection" and not math.isnan(r.chosen_pos_error_m) and not math.isnan(r.no_int_random_err_m)
    ]
    if not pairs:
        return {"count": 0, "fallback_mean": float("nan"), "random_mean": float("nan"),
                "mean_diff_random_minus_fallback": float("nan"), "w_stat": float("nan"), "p_value": float("nan")}
    fallback_arr = np.array([p[0] for p in pairs])
    random_arr = np.array([p[1] for p in pairs])
    diff = random_arr - fallback_arr  # positive => fallback better
    w_stat = float("nan")
    p_value = float("nan")
    if wilcoxon is not None:
        try:
            w_raw, p_raw = wilcoxon(diff, zero_method="pratt", alternative="greater", method="approx")
            w_stat = float(w_raw)
            p_value = float(p_raw)
        except Exception:
            pass
    return {
        "count": float(len(pairs)),
        "fallback_mean": float(np.mean(fallback_arr)),
        "random_mean": float(np.mean(random_arr)),
        "mean_diff_random_minus_fallback": float(np.mean(diff)),
        "w_stat": w_stat,
        "p_value": p_value,
    }


def compute_kinematic_mcnemar_vs_random(results: List[SimResult]) -> Dict[str, float]:
    """McNemar test comparing kinematic fallback vs random intersection choice."""
    a = b = c = d = 0
    for r in results:
        if r.kinematic_correct is None or r.kinematic_random_correct is None:
            continue
        mc = r.kinematic_correct
        rc = r.kinematic_random_correct
        if mc and rc:
            a += 1
        elif mc and not rc:
            b += 1
        elif not mc and rc:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    discordant = b + c
    method_acc = (a + b) / n if n > 0 else float("nan")
    baseline_acc = (a + c) / n if n > 0 else float("nan")
    chi2_cc = ((abs(b - c) - 1.0) ** 2) / discordant if discordant > 0 else float("nan")
    p_exact = float("nan")
    if binomtest is not None and discordant > 0:
        try:
            p_exact = float(binomtest(k=b, n=discordant, p=0.5, alternative="greater").pvalue)
        except Exception:
            pass
    return {
        "count": float(n), "a": float(a), "b": float(b), "c": float(c), "d": float(d),
        "discordant": float(discordant), "method_accuracy": method_acc, "baseline_accuracy": baseline_acc,
        "chi2_cc": chi2_cc, "p_exact_one_sided": p_exact,
    }


def compute_kinematic_position_error(results: List[SimResult]) -> Dict[str, float]:
    """Positioning error stats for kinematically chosen intersections."""
    errs = [r.kinematic_chosen_pos_error_m for r in results
            if r.kinematic_correct is not None and not math.isnan(r.kinematic_chosen_pos_error_m)]
    if not errs:
        return {"count": 0, "mean": float("nan"), "p90": float("nan")}
    errs_s = sorted(errs)
    n = len(errs_s)
    mean_v = sum(errs_s) / n
    p = 0.90
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    p90_v = errs_s[lo] if lo == hi else errs_s[lo] * (1.0 - (pos - lo)) + errs_s[hi] * (pos - lo)
    return {"count": float(n), "mean": mean_v, "p90": p90_v}


def aggregate_exhaustive(results: List[SimResult]) -> List[Dict[str, object]]:
    """Build exhaustive table over all 4^4 GT/measured bin combinations."""
    stats: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in results:
        key = (r.gt_bin_a1, r.gt_bin_a2, r.meas_bin_a1, r.meas_bin_a2)
        stats[key]["total"] += 1
        stats[key][r.outcome] += 1

    rows: List[Dict[str, object]] = []
    for gt1, gt2, m1, m2 in product(ANGLE_BINS, ANGLE_BINS, ANGLE_BINS, ANGLE_BINS):
        key = (gt1, gt2, m1, m2)
        s = stats.get(key, {})
        total = int(s.get("total", 0))
        correct = int(s.get("correct", 0))
        wrong = int(s.get("wrong", 0))
        no_inter = int(s.get("no_intersection", 0))
        deg = int(s.get("degenerate", 0))

        if total > 0:
            correct_rate = correct / total
            wrong_rate = wrong / total
            no_inter_rate = no_inter / total
            deg_rate = deg / total
            observed = "yes"
        else:
            correct_rate = wrong_rate = no_inter_rate = deg_rate = 0.0
            observed = "no"

        rows.append(
            {
                "gt_bin_a1": gt1,
                "gt_bin_a2": gt2,
                "meas_bin_a1": m1,
                "meas_bin_a2": m2,
                "observed_in_simulation": observed,
                "samples": total,
                "correct": correct,
                "wrong": wrong,
                "no_intersection": no_inter,
                "degenerate": deg,
                "correct_rate": round(correct_rate, 4),
                "wrong_rate": round(wrong_rate, 4),
                "no_intersection_rate": round(no_inter_rate, 4),
                "degenerate_rate": round(deg_rate, 4),
            }
        )

    return rows


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: List[Dict[str, object]], top_k: int = 15) -> None:
    observed = [r for r in rows if r["observed_in_simulation"] == "yes" and r["samples"] > 0]
    observed_sorted = sorted(observed, key=lambda r: (r["wrong_rate"], r["no_intersection_rate"]), reverse=True)

    print("\nTop high-risk observed combinations (by wrong_rate, then no_intersection_rate):")
    print("gt(a1), gt(a2), meas(a1), meas(a2), samples, wrong_rate, no_intersection_rate")
    for r in observed_sorted[:top_k]:
        print(
            f"{r['gt_bin_a1']}, {r['gt_bin_a2']}, {r['meas_bin_a1']}, {r['meas_bin_a2']}, "
            f"{r['samples']}, {r['wrong_rate']:.4f}, {r['no_intersection_rate']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-anchor boundary-case simulator")
    p.add_argument("--samples", type=int, default=120000, help="Number of Monte Carlo samples")
    p.add_argument(
        "--scenario",
        type=str,
        default="wrap_noise",
        choices=["ideal", "wrap", "wrap_noise", "ideal_noise", "flip_wrap"],
        help="Measurement behavior model",
    )
    p.add_argument(
        "--anchor-mode",
        type=str,
        default="random",
        choices=["random", "fixed"],
        help="Anchor placement mode",
    )
    p.add_argument(
        "--fixed-anchor-distance",
        type=float,
        default=1.93,
        help="Distance between anchors when --anchor-mode fixed (m)",
    )
    p.add_argument(
        "--min-anchor-separation",
        type=float,
        default=0.5,
        help="Minimum anchor-to-anchor separation when --anchor-mode random (m)",
    )
    p.add_argument("--range-sigma", type=float, default=0.05, help="Std dev of range noise (m)")
    p.add_argument("--range-mu", type=float, default=0.0, help="Mean of range noise (m)")
    p.add_argument("--angle-sigma", type=float, default=8.0, help="Std dev of angle noise (deg)")
    p.add_argument("--angle-mu", type=float, default=0.0, help="Mean of angle noise (deg)")
    p.add_argument("--min-distance", type=float, default=0.5, help="Minimum true distance to each anchor (m)")
    p.add_argument("--max-distance", type=float, default=40.0, help="Maximum true distance to each anchor (m)")
    p.add_argument(
        "--exclude-front-behind-misreports",
        action="store_true",
        help="Exclude cases where |GT angle| < 90 but |measured angle| > 90",
    )
    p.add_argument(
        "--allow-no-fb-hemisphere-crossing",
        action="store_true",
        help="For scenario=ideal_noise, allow noisy angle to cross front/back boundary",
    )
    p.add_argument(
        "--kinematic-fallback",
        action="store_true",
        help="Evaluate kinematic-consistency fallback for FB ambiguity in each trial",
    )
    p.add_argument("--kinematic-step-mu", type=float, default=0.7, help="Mean step size for kinematic fallback (m)")
    p.add_argument("--kinematic-prev-pos-sigma", type=float, default=0.5, help="Trilateration noise σ on prior position estimate (m)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out",
        type=str,
        default="two_anchor_exhaustive_cases.csv",
        help="CSV output path",
    )
    p.add_argument(
        "--stats-out",
        type=str,
        default="",
        help="Optional JSON path for run-level summary statistics",
    )
    p.add_argument("--top-k", type=int, default=15, help="How many high-risk combos to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_distance < 0 or args.max_distance <= 0 or args.min_distance > args.max_distance:
        raise ValueError("Invalid distance bounds: require 0 <= min-distance <= max-distance")
    if args.min_anchor_separation < 0:
        raise ValueError("Invalid min-anchor-separation: must be >= 0")

    print("Running simulation...")
    print(
        f"scenario={args.scenario}, samples={args.samples}, "
        f"anchor_mode={args.anchor_mode}, fixed_anchor_distance={args.fixed_anchor_distance}, "
        f"min_anchor_separation={args.min_anchor_separation}, "
        f"range_sigma={args.range_sigma}, range_mu={args.range_mu}, "
        f"angle_sigma={args.angle_sigma}, angle_mu={args.angle_mu}, "
        f"min_distance={args.min_distance}, max_distance={args.max_distance}, "
        f"exclude_front_behind_misreports={args.exclude_front_behind_misreports}, "
        f"kinematic_fallback={args.kinematic_fallback}"
    )
    if args.kinematic_fallback:
        print(f"  step_mu={args.kinematic_step_mu}, prev_pos_sigma={args.kinematic_prev_pos_sigma}")

    results = run_simulation(
        samples=args.samples,
        scenario=args.scenario,
        anchor_mode=args.anchor_mode,
        fixed_anchor_distance=args.fixed_anchor_distance,
        min_anchor_separation_m=args.min_anchor_separation,
        range_sigma_m=args.range_sigma,
        range_mu_m=args.range_mu,
        angle_sigma_deg=args.angle_sigma,
        angle_mu_deg=args.angle_mu,
        seed=args.seed,
        min_distance_m=args.min_distance,
        max_distance_m=args.max_distance,
        exclude_front_to_behind_misreports=args.exclude_front_behind_misreports,
        enforce_no_fb_hemisphere_consistency=not args.allow_no_fb_hemisphere_crossing,
        kinematic_fallback=args.kinematic_fallback,
        kinematic_step_mu=args.kinematic_step_mu,
        kinematic_prev_pos_sigma=args.kinematic_prev_pos_sigma,
    )

    rows = aggregate_exhaustive(results)
    write_csv(rows, args.out)

    print(f"\nExhaustive table written to: {args.out}")
    print(f"Total rows (4^4 combinations): {len(rows)}")

    observed = sum(1 for r in rows if r["observed_in_simulation"] == "yes")
    print(f"Observed combinations in simulation: {observed}/{len(rows)}")

    rates = compute_requested_success_rates(results)
    print("\nRequested success rates:")
    print(
        f"1) all cases: {rates['all_cases']:.4f} "
        f"(n={int(rates['count_total'])})"
    )
    print(
        f"2) only when intersections exist: {rates['intersections_exist']:.4f} "
        f"(n={int(rates['count_with_intersections'])})"
    )
    print(
        f"3) only when both anchors are in front (|GT angle| < 90): {rates['both_anchors_front']:.4f} "
        f"(n={int(rates['count_both_front'])})"
    )

    fb = compute_front_back_breakdown(results)
    print("\nFront/behind breakdown (by GT anchor categories):")
    print(
        f"1) both anchors in front: {fb['both_front']['success_rate']:.4f} "
        f"(n={int(fb['both_front']['count'])})"
    )
    print(
        f"2) one anchor in front, one anchor behind: {fb['one_front_one_behind']['success_rate']:.4f} "
        f"(n={int(fb['one_front_one_behind']['count'])})"
    )
    print(
        f"3) both anchors behind: {fb['both_behind']['success_rate']:.4f} "
        f"(n={int(fb['both_behind']['count'])})"
    )

    pos_stats = compute_position_error_stats(results)
    pos_all_stats = compute_position_error_stats_all(results)
    pos_fallback_stats = compute_position_error_stats(results, include_outcomes=("no_intersection",))
    ci_half = (pos_stats["ci95_hi"] - pos_stats["ci95_lo"]) / 2.0 if not math.isnan(pos_stats["mean"]) else float("nan")
    print("\nPositioning error on successful disambiguations:")
    print(
        f"mean={pos_stats['mean']:.4f}, std={pos_stats['std']:.4f}, "
        f"95% CI={pos_stats['mean']:.4f} ± {ci_half:.4f}, "
        f"P90={pos_stats['p90']:.4f}, n={int(pos_stats['count'])}"
    )

    wilcoxon_stats = compute_wilcoxon_vs_random(results)
    print("\nWilcoxon paired-rank test vs random-intersection baseline (ambiguous cases):")
    print(
        f"n={int(wilcoxon_stats['count'])}, heuristic_mean={wilcoxon_stats['heuristic_mean']:.4f}, "
        f"random_mean={wilcoxon_stats['random_mean']:.4f}, "
        f"mean_diff(random-heuristic)={wilcoxon_stats['mean_diff_random_minus_heuristic']:.4f}, "
        f"W={wilcoxon_stats['w_stat']:.4f}, p={wilcoxon_stats['p_value']:.6g}"
    )

    mcnemar_stats = compute_mcnemar_vs_random(results)
    print("\nMcNemar test vs random-intersection baseline on intersection-choice accuracy:")
    print(
        f"n={int(mcnemar_stats['count'])}, "
        f"acc_method={mcnemar_stats['method_accuracy']:.4f}, "
        f"acc_random={mcnemar_stats['baseline_accuracy']:.4f}, "
        f"a={int(mcnemar_stats['a'])}, b={int(mcnemar_stats['b'])}, "
        f"c={int(mcnemar_stats['c'])}, d={int(mcnemar_stats['d'])}, "
        f"chi2_cc={mcnemar_stats['chi2_cc']:.4f}, "
        f"p_exact_one_sided={mcnemar_stats['p_exact_one_sided']:.6g}"
    )

    kinematic_stats = {}
    no_int_wilcoxon_stats = compute_no_int_wilcoxon_vs_random(results)
    kin_mcnemar_stats = compute_kinematic_mcnemar_vs_random(results)
    kin_pos_err_stats = compute_kinematic_position_error(results)
    if args.kinematic_fallback:
        kinematic_stats = compute_kinematic_fallback_stats(results)
        if kinematic_stats:
            print("\n--- Kinematic-Consistency Fallback ---")
            print(f"  Total trials:               {kinematic_stats['n']}")
            print(f"  Overall accuracy:           {100*kinematic_stats['accuracy']:.1f}%")
            print(f"  When NOT crossing baseline: {100*kinematic_stats['accuracy_no_cross']:.1f}%")
            if kinematic_stats['n'] > 0:
                n_cr = int(kinematic_stats['pct_crossing'] * kinematic_stats['n'] / 100)
                print(f"  When crossing baseline:     {100*kinematic_stats['accuracy_crossing']:.1f}% (n={n_cr})")
            print(f"  Baseline crossing rate:     {kinematic_stats['pct_crossing']:.1f}%")
            print(f"  Avg intersection separation:{kinematic_stats['avg_inter_sep_m']:.2f} m")
            print(f"  Avg prev position error:    {kinematic_stats['avg_prev_error_m']:.2f} m")
            print(f"  Avg dist to correct inter:  {kinematic_stats['avg_dist_correct_m']:.2f} m")
            print(f"  Avg dist to wrong inter:    {kinematic_stats['avg_dist_wrong_m']:.2f} m")
            print("  By intersection separation:")
            for label, s in kinematic_stats.get('by_intersection_separation', {}).items():
                print(f"    {label:>8s}: {100*s['acc']:.1f}% (n={s['n']}, {s['pct']:.1f}% of trials)")

    if args.stats_out:
        payload = {
            "scenario": args.scenario,
            "samples": args.samples,
            "range_sigma_m": args.range_sigma,
            "range_mu_m": args.range_mu,
            "angle_sigma_deg": args.angle_sigma,
            "angle_mu_deg": args.angle_mu,
            "exclude_front_behind_misreports": bool(args.exclude_front_behind_misreports),
            "success_rates": rates,
            "front_back_breakdown": fb,
            "position_error_success_only": pos_stats,
            "position_error_all": pos_all_stats,
            "position_error_fallback": pos_fallback_stats,
            "wilcoxon_vs_random": wilcoxon_stats,
            "mcnemar_vs_random": mcnemar_stats,
            "no_int_fallback_wilcoxon_vs_random": no_int_wilcoxon_stats,
            "kinematic_fallback_mcnemar_vs_random": kin_mcnemar_stats,
            "kinematic_fallback_position_error": kin_pos_err_stats,
        }
        if kinematic_stats:
            payload["kinematic_fallback"] = kinematic_stats
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Summary stats written to: {args.stats_out}")

    print_summary(rows, top_k=args.top_k)


if __name__ == "__main__":
    main()
