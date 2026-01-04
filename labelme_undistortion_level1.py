#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

# =============================
# CONFIG
# =============================
ROOT = "fencing_piste_sample_data"  # change as needed. Shi Jan. 4, 2026

# Visualization outputs (lines/points overlaid)
OUT_VIZ_DIR = "viz_level1"

# ---- NEW: export undistorted artifacts ----
OUT_UNDIST_IMG_DIR = "undistorted_images"          # undistorted (or raw) images written here
OUT_UNDIST_LABELME_DIR = "undistorted_labelme_json_files"     # Labelme JSONs with undistorted points written here
OUT_UNDIST_POINTS_DIR = "undistorted_points_txt"   # per-image txt of undistorted points (label x y)

IMG_EXTS = {".jpg", ".png", ".jpeg"}

# ---- master switch: apply distortion correction or not ----
# If False: k1 is forced to 0 and images/points are written "raw".
APPLY_DISTORTION_CORRECTION = True

# ---- Optional: use vanishing point as constraint during optimization ----
# If True: optimize (k1, vp_x_n, vp_y_n) per camera group and fit each line through that VP.
USE_VP_CONSTRAINT = False  # set True if you want VP-constrained fitting

# Bounds
K1_MIN, K1_MAX = -0.5, 0.5
VP_BOUND = 5.0  # VP bounds in normalized coords (can be far outside image)

# Drawing parameters
UNDISTORT_ITERATIONS = 8
POINT_RADIUS = 4
LINE_THICKNESS = 2
OBB_THICKNESS = 2

# ---- Vanishing point visualization ----
DRAW_VANISHING_POINT = True
VP_MIN_LINES = 2
VP_DRAW_IF_OUTSIDE = True
VP_MARKER_RADIUS = 8
VP_TEXT_SCALE = 0.6
VP_TEXT_THICKNESS = 2

# Colors (BGR)
COLOR_POINTS = (0, 0, 255)        # red: points (undistorted if enabled)
COLOR_NEAR_LINE = (255, 0, 0)     # blue: fitted near line
COLOR_FAR_LINE = (0, 255, 255)    # yellow: fitted far line
COLOR_VP = (255, 0, 255)          # magenta: vanishing point

# If True, compute VP only from the drawn lines (near+far) if no constrained VP is available
COMPUTE_VP_FROM_DRAWN_LINES = True


# =============================
# RADIAL UNDISTORTION (k1 only)
# =============================
def normalize_xy(xy: np.ndarray, w: int, h: int):
    cx, cy = w / 2.0, h / 2.0
    s = max(w, h) / 2.0
    return (xy - np.array([cx, cy], dtype=np.float64)) / s, cx, cy, s


def denormalize_xy(xy_n: np.ndarray, cx: float, cy: float, s: float):
    return xy_n * s + np.array([cx, cy], dtype=np.float64)


def undistort_radial(xy: np.ndarray, w: int, h: int, k1: float):
    """
    Forward radial model on points (normalized):
        x_u = x * (1 + k1*r^2)
        y_u = y * (1 + k1*r^2)
    Used for straightness scoring and inversion inside image remap iterations.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return xy.reshape(0, 2)

    xy_n, cx, cy, s = normalize_xy(xy, w, h)
    x, y = xy_n[:, 0], xy_n[:, 1]
    r2 = x * x + y * y
    scale = 1.0 + k1 * r2
    xu = x * scale
    yu = y * scale
    return denormalize_xy(np.stack([xu, yu], axis=1), cx, cy, s)


def undistort_image_radial(img: np.ndarray, k1: float, iterations: int = 8):
    """
    Undistorted output grid U -> solve for distorted sampling coords D such that
    undistort_radial(D) ~= U, then remap(input=distorted img, map=D).
    """
    h, w = img.shape[:2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    U = np.column_stack((uu.ravel(), vv.ravel())).astype(np.float64)

    D = U.copy()
    for _ in range(iterations):
        U_hat = undistort_radial(D, w, h, k1)
        D -= (U_hat - U)

    mapx = D[:, 0].reshape(h, w).astype(np.float32)
    mapy = D[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# =============================
# LINE FITTING
# =============================
def fit_line_tls(pts: np.ndarray):
    """
    Total least squares line: returns (centroid c, unit direction d).
    Line = c + t*d
    """
    c = pts.mean(axis=0)
    X = pts - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    return c, d


def line_residuals(pts: np.ndarray):
    """
    Orthogonal residuals to best-fit line (TLS).
    Returns per-point distances to the best-fit infinite line.
    """
    if len(pts) < 3:
        return np.zeros((0,), dtype=np.float64)
    c, d = fit_line_tls(pts)
    v = pts - c
    proj = (v @ d)[:, None] * d[None, :]
    ortho = v - proj
    return np.linalg.norm(ortho, axis=1)


def vp_norm_to_pixel(vp_n: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    vp_n: (2,) in normalized coords (same normalize_xy system)
    returns vp pixel coords (2,)
    """
    cx, cy = w / 2.0, h / 2.0
    s = max(w, h) / 2.0
    return np.array([vp_n[0] * s + cx, vp_n[1] * s + cy], dtype=np.float64)


def fit_line_through_point_tls(pts: np.ndarray, p0: np.ndarray):
    """
    Fit best line direction d constrained to pass through point p0.
    Solve on vectors v_i = (p_i - p0). Best direction is principal component of v_i.
    Returns (p0, d).
    """
    if len(pts) < 2:
        return p0, np.array([1.0, 0.0], dtype=np.float64)

    V = pts - p0[None, :]
    if np.linalg.norm(V) < 1e-9:
        return p0, np.array([1.0, 0.0], dtype=np.float64)

    _, _, vt = np.linalg.svd(V, full_matrices=False)
    d = vt[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    return p0, d


def line_residuals_through_point(pts: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """
    Orthogonal distances to best line constrained to pass through p0.
    """
    if len(pts) < 3:
        return np.zeros((0,), dtype=np.float64)

    c, d = fit_line_through_point_tls(pts, p0)
    v = pts - c
    proj = (v @ d)[:, None] * d[None, :]
    ortho = v - proj
    return np.linalg.norm(ortho, axis=1)


# =============================
# LEVEL-1 RESIDUALS (k1-only + optional VP constraint)
# =============================
def residuals_level1(params, items):
    """
    If USE_VP_CONSTRAINT:
      params = [k1, vp_x_n, vp_y_n] in normalized coords.
      Each near/far instance set is fitted to a line forced THROUGH vp_pix.

    Else:
      params = [k1]
      Each set fitted with standard TLS line.
    """
    if USE_VP_CONSTRAINT:
        k1 = float(params[0])
        vp_n = np.array([float(params[1]), float(params[2])], dtype=np.float64)
    else:
        k1 = float(params[0])
        vp_n = None

    res = []
    for w, h, near_sets, far_sets in items:
        vp_pix = vp_norm_to_pixel(vp_n, w, h) if USE_VP_CONSTRAINT else None

        for near in near_sets:
            if len(near) >= 3:
                near_u = undistort_radial(near, w, h, k1)
                if USE_VP_CONSTRAINT:
                    res.append(line_residuals_through_point(near_u, vp_pix))
                else:
                    res.append(line_residuals(near_u))

        for far in far_sets:
            if len(far) >= 3:
                far_u = undistort_radial(far, w, h, k1)
                if USE_VP_CONSTRAINT:
                    res.append(line_residuals_through_point(far_u, vp_pix))
                else:
                    res.append(line_residuals(far_u))

    return np.concatenate(res) if res else np.zeros((0,), dtype=np.float64)


# =============================
# DRAW / VP HELPERS
# =============================
def draw_infinite_line(img: np.ndarray, c: np.ndarray, d: np.ndarray, color, thickness: int):
    """
    Draw line c + t*d across the image by intersecting with image borders.
    """
    h, w = img.shape[:2]
    eps = 1e-12
    points = []

    if abs(d[0]) > eps:
        t = (0 - c[0]) / d[0]
        y = c[1] + t * d[1]
        if 0 <= y <= h - 1:
            points.append((0, int(round(y))))
        t = ((w - 1) - c[0]) / d[0]
        y = c[1] + t * d[1]
        if 0 <= y <= h - 1:
            points.append((w - 1, int(round(y))))

    if abs(d[1]) > eps:
        t = (0 - c[1]) / d[1]
        x = c[0] + t * d[0]
        if 0 <= x <= w - 1:
            points.append((int(round(x)), 0))
        t = ((h - 1) - c[1]) / d[1]
        x = c[0] + t * d[0]
        if 0 <= x <= w - 1:
            points.append((int(round(x)), h - 1))

    if len(points) < 2:
        return

    uniq = []
    for p in points:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) < 2:
        return

    cv2.line(img, uniq[0], uniq[1], color, thickness)


def line_to_abc(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Convert line (c + t*d) to ax+by+c=0 with ||(a,b)||=1
    """
    n = np.array([-d[1], d[0]], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    n /= nn
    a, b = n[0], n[1]
    cc = -(a * c[0] + b * c[1])
    return np.array([a, b, cc], dtype=np.float64)


def estimate_vanishing_point(lines_abc: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Solve for vp (x,y) that minimizes a_i x + b_i y + c_i = 0 across lines.
    Robust Huber least squares.
    """
    if len(lines_abc) < VP_MIN_LINES:
        return None

    L = np.stack(lines_abc, axis=0).astype(np.float64)
    A = L[:, :2]
    if np.linalg.matrix_rank(A) < 2:
        return None

    def vp_res(p):
        x, y = float(p[0]), float(p[1])
        return (L[:, 0] * x + L[:, 1] * y + L[:, 2])

    p0, *_ = np.linalg.lstsq(A, -L[:, 2], rcond=None)
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)

    opt = least_squares(vp_res, x0=p0, loss="huber", f_scale=5.0, max_nfev=200)
    return opt.x.astype(np.float64)


def draw_vanishing_point(img: np.ndarray, vp_xy: np.ndarray, color=COLOR_VP):
    h, w = img.shape[:2]
    x, y = float(vp_xy[0]), float(vp_xy[1])
    inside = (0 <= x < w) and (0 <= y < h)
    if not inside and not VP_DRAW_IF_OUTSIDE:
        return

    xd = int(round(min(max(x, 0.0), w - 1.0)))
    yd = int(round(min(max(y, 0.0), h - 1.0)))

    cv2.circle(img, (xd, yd), VP_MARKER_RADIUS, color, 2)
    cv2.drawMarker(img, (xd, yd), color, markerType=cv2.MARKER_CROSS, markerSize=VP_MARKER_RADIUS * 2, thickness=2)
    cv2.putText(
        img,
        f"VP=({x:.1f},{y:.1f})",
        (max(0, xd + 10), max(20, yd - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        VP_TEXT_SCALE,
        color,
        VP_TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def group_key(name: str):
    stem = Path(name).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def find_image(jp: Path, data: Dict[str, Any]) -> Optional[Path]:
    ip = data.get("imagePath")
    if ip:
        p = (jp.parent / ip)
        if p.exists():
            return p
    for ext in IMG_EXTS:
        p = jp.with_suffix(ext)
        if p.exists():
            return p
    return None


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception:
        # fall back to raw copy
        dst.write_bytes(src.read_bytes())


# =============================
# PARSING HELPERS (multi-instance near/far via group_id)
# =============================
def extract_instance_sets(data: Dict[str, Any]) -> Tuple[
    Dict[Any, Dict[str, List]],
    List[np.ndarray],
    List[np.ndarray],
]:
    inst: Dict[Any, Dict[str, list]] = defaultdict(lambda: {"pts": [], "near": [], "far": []})

    for sh in data.get("shapes", []):
        if sh.get("shape_type") != "point":
            continue
        label = sh.get("label", "")
        x, y = sh["points"][0]
        gid = sh.get("group_id", 0)

        inst[gid]["pts"].append((float(x), float(y), label))
        if label.endswith("_near"):
            inst[gid]["near"].append((float(x), float(y)))
        elif label.endswith("_far"):
            inst[gid]["far"].append((float(x), float(y)))

    near_sets = [np.array(v["near"], dtype=np.float64) for v in inst.values() if len(v["near"]) > 0]
    far_sets = [np.array(v["far"], dtype=np.float64) for v in inst.values() if len(v["far"]) > 0]
    return inst, near_sets, far_sets


# =============================
# MAIN
# =============================
def main():
    root = Path(ROOT)
    viz_dir = Path(OUT_VIZ_DIR)
    viz_dir.mkdir(parents=True, exist_ok=True)

    und_img_dir = Path(OUT_UNDIST_IMG_DIR)
    und_img_dir.mkdir(parents=True, exist_ok=True)

    und_labelme_dir = Path(OUT_UNDIST_LABELME_DIR)
    und_labelme_dir.mkdir(parents=True, exist_ok=True)

    und_points_dir = Path(OUT_UNDIST_POINTS_DIR)
    und_points_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No json files under {root.resolve()}")

    # 1) Collect near/far sets for calibration per camera group
    cam_groups: Dict[str, list] = defaultdict(list)
    for jp in json_files:
        data = json.loads(jp.read_text())
        img_path = find_image(jp, data)
        if img_path is None:
            continue

        w = int(data["imageWidth"])
        h = int(data["imageHeight"])
        gk = group_key(data.get("imagePath", img_path.name))

        _, near_sets, far_sets = extract_instance_sets(data)
        if near_sets or far_sets:
            cam_groups[gk].append((w, h, near_sets, far_sets))

    # 2) Solve per camera group
    # Store dict per group: {"k1": float, "vp_n": (float,float) or None}
    cam_params: Dict[str, Dict[str, Any]] = {}

    if not APPLY_DISTORTION_CORRECTION:
        for gk in cam_groups.keys():
            cam_params[gk] = {"k1": 0.0, "vp_n": None}
        print("[LEVEL-1] distortion correction disabled -> using k1=0 for all groups")
    else:
        for gk, items in cam_groups.items():
            usable = any(
                any(len(ns) >= 3 for ns in near_sets) or any(len(fs) >= 3 for fs in far_sets)
                for _, _, near_sets, far_sets in items
            )
            if not usable:
                cam_params[gk] = {"k1": 0.0, "vp_n": None}
                print(f"[LEVEL-1] group={gk} skipped (not enough points) -> k1=0")
                continue

            if USE_VP_CONSTRAINT:
                opt = least_squares(
                    residuals_level1,
                    x0=np.array([0.0, 0.0, 0.0], dtype=np.float64),  # k1, vp_x_n, vp_y_n
                    bounds=(
                        np.array([K1_MIN, -VP_BOUND, -VP_BOUND], dtype=np.float64),
                        np.array([K1_MAX,  VP_BOUND,  VP_BOUND], dtype=np.float64),
                    ),
                    args=(items,),
                    loss="huber",
                    f_scale=2.0,
                    max_nfev=300,
                )
                cam_params[gk] = {"k1": float(opt.x[0]), "vp_n": (float(opt.x[1]), float(opt.x[2]))}
                print(f"[LEVEL-1+VP] group={gk}  k1={opt.x[0]:.6g}  vp_n=({opt.x[1]:.3g},{opt.x[2]:.3g})")
            else:
                opt = least_squares(
                    residuals_level1,
                    x0=np.array([0.0], dtype=np.float64),
                    bounds=(np.array([K1_MIN], dtype=np.float64), np.array([K1_MAX], dtype=np.float64)),
                    args=(items,),
                    loss="huber",
                    f_scale=2.0,
                    max_nfev=200,
                )
                cam_params[gk] = {"k1": float(opt.x[0]), "vp_n": None}
                print(f"[LEVEL-1] group={gk}  k1={opt.x[0]:.6g}")

    # 3) Per-image:
    #   - write undistorted (or raw) image to OUT_UNDIST_IMG_DIR
    #   - write undistorted points Labelme JSON to OUT_UNDIST_LABELME_DIR
    #   - write a txt file of corresponding points to OUT_UNDIST_POINTS_DIR
    #   - write visualization to OUT_VIZ_DIR
    for jp in json_files:
        data = json.loads(jp.read_text())
        img_path = find_image(jp, data)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        gk = group_key(data.get("imagePath", img_path.name))

        entry = cam_params.get(gk, {"k1": 0.0, "vp_n": None})
        k1 = float(entry["k1"]) if APPLY_DISTORTION_CORRECTION else 0.0

        vp_pix = None
        if APPLY_DISTORTION_CORRECTION and USE_VP_CONSTRAINT and entry.get("vp_n") is not None:
            vp_n = np.array(entry["vp_n"], dtype=np.float64)
            vp_pix = vp_norm_to_pixel(vp_n, w_img, h_img)

        # Build the undistorted (or raw) image
        if APPLY_DISTORTION_CORRECTION and abs(k1) > 0.0:
            und_img = undistort_image_radial(img, k1, iterations=UNDISTORT_ITERATIONS)
        else:
            und_img = img.copy()

        # ---- Save undistorted image (same extension as source if possible) ----
        ext = img_path.suffix.lower()
        if ext not in IMG_EXTS:
            ext = ".jpg"
        suffix = "und" if APPLY_DISTORTION_CORRECTION else "raw"
        und_name = f"{jp.stem}_{suffix}{ext}"
        und_img_path = und_img_dir / und_name
        cv2.imwrite(str(und_img_path), und_img)

        # ---- Compute undistorted corresponding points for all point shapes ----
        # We will update a copy of Labelme JSON: for each point shape, replace coords with undistorted coords.
        new_data = dict(data)  # shallow copy
        new_shapes = []
        points_txt_lines: List[str] = []

        for sh in data.get("shapes", []):
            if sh.get("shape_type") != "point":
                # Keep non-point shapes unchanged (cannot reliably warp polygons here)
                new_shapes.append(sh)
                continue

            label = sh.get("label", "")
            x, y = sh["points"][0]
            xy = np.array([[float(x), float(y)]], dtype=np.float64)

            if APPLY_DISTORTION_CORRECTION and abs(k1) > 0.0:
                xy_u = undistort_radial(xy, w_img, h_img, k1)[0]
            else:
                xy_u = xy[0]

            sh2 = dict(sh)
            sh2["points"] = [[float(xy_u[0]), float(xy_u[1])]]
            new_shapes.append(sh2)

            # Corresponding-point record: label x y (undistorted)
            points_txt_lines.append(f"{label}\t{xy_u[0]:.3f}\t{xy_u[1]:.3f}")

        new_data["shapes"] = new_shapes

        # Update imagePath to point to the newly written image file name (relative)
        new_data["imagePath"] = und_name

        # imageData can make JSON huge; remove it if present to keep files light
        if "imageData" in new_data:
            new_data["imageData"] = None

        # Keep width/height unchanged
        new_data["imageWidth"] = int(w_img)
        new_data["imageHeight"] = int(h_img)

        # Save new Labelme JSON
        und_json_path = und_labelme_dir / f"{jp.stem}_{suffix}.json"
        und_json_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2))

        # Save corresponding points txt
        pts_txt_path = und_points_dir / f"{jp.stem}_{suffix}.txt"
        pts_txt_path.write_text("\n".join(points_txt_lines) + ("\n" if points_txt_lines else ""))

        # ---- Visualization: draw lines + points on the undistorted (or raw) image ----
        canvas = und_img.copy()
        inst, _, _ = extract_instance_sets(data)

        lines_for_vp: List[np.ndarray] = []

        for _, v in inst.items():
            # Near
            if len(v["near"]) >= 3:
                near_xy = np.array(v["near"], dtype=np.float64)
                near_draw = undistort_radial(near_xy, w_img, h_img, k1) if (APPLY_DISTORTION_CORRECTION and abs(k1) > 0.0) else near_xy
                if USE_VP_CONSTRAINT and (vp_pix is not None):
                    c, d = fit_line_through_point_tls(near_draw, vp_pix)
                else:
                    c, d = fit_line_tls(near_draw)
                draw_infinite_line(canvas, c, d, COLOR_NEAR_LINE, LINE_THICKNESS)
                if COMPUTE_VP_FROM_DRAWN_LINES:
                    lines_for_vp.append(line_to_abc(c, d))

            # Far
            if len(v["far"]) >= 3:
                far_xy = np.array(v["far"], dtype=np.float64)
                far_draw = undistort_radial(far_xy, w_img, h_img, k1) if (APPLY_DISTORTION_CORRECTION and abs(k1) > 0.0) else far_xy
                if USE_VP_CONSTRAINT and (vp_pix is not None):
                    c, d = fit_line_through_point_tls(far_draw, vp_pix)
                else:
                    c, d = fit_line_tls(far_draw)
                draw_infinite_line(canvas, c, d, COLOR_FAR_LINE, LINE_THICKNESS)
                if COMPUTE_VP_FROM_DRAWN_LINES:
                    lines_for_vp.append(line_to_abc(c, d))

        # Draw points (using undistorted points)
        for sh in new_shapes:
            if sh.get("shape_type") != "point":
                continue
            lab = sh.get("label", "")
            x_u, y_u = sh["points"][0]
            cv2.circle(canvas, (int(round(x_u)), int(round(y_u))), POINT_RADIUS, COLOR_POINTS, -1)
            cv2.putText(
                canvas,
                lab,
                (int(round(x_u)) + 4, int(round(y_u)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_POINTS,
                1,
                lineType=cv2.LINE_AA,
            )

        # Draw VP if requested
        if DRAW_VANISHING_POINT:
            vp_to_draw = None
            if USE_VP_CONSTRAINT and (vp_pix is not None):
                vp_to_draw = vp_pix
            elif COMPUTE_VP_FROM_DRAWN_LINES:
                valid = [L for L in lines_for_vp if np.linalg.norm(L[:2]) > 1e-12]
                vp_est = estimate_vanishing_point(valid)
                vp_to_draw = vp_est

            if vp_to_draw is not None:
                draw_vanishing_point(canvas, vp_to_draw, COLOR_VP)

        viz_path = viz_dir / f"{jp.stem}_level1_{suffix}.jpg"
        cv2.imwrite(str(viz_path), canvas)

    print("Done.")
    print(f"Undistorted images: {und_img_dir.resolve()}")
    print(f"Undistorted labelme json: {und_labelme_dir.resolve()}")
    print(f"Undistorted points txt: {und_points_dir.resolve()}")
    print(f"Visualization: {viz_dir.resolve()}")


if __name__ == "__main__":
    main()
