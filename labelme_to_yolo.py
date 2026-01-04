#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
# Option A (recommended if you already created undistorted outputs):
#   - JSON_ROOT points to undistorted Labelme JSON folder
#   - IMAGES_ROOT points to undistorted image folder
#
# Option B:
#   - JSON_ROOT points to your original Labelme dataset folder
#   - IMAGES_ROOT can be None (script will look via imagePath or same-stem)

USE_PREUNDISTORTED = True
JSON_ROOT = "undistorted_labelme_json_files"       # e.g., labelme_undistorted (from prior script)
IMAGES_ROOT = "undistorted_images"      # e.g., undistorted_images (from prior script)

# If you want to use the original Labelme data directly:
# USE_PREUNDISTORTED = False
# JSON_ROOT = "labelme_data"
# IMAGES_ROOT = None

OUT_ROOT = "yolo_obb_dataset"

# Splits
VAL_RATIO = 0.2
RANDOM_SEED = 123

# Image extensions to probe if imagePath fails
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# Label handling
STRIP_SUFFIXES = ["_near", "_far"]   # remove these from per-point labels to get class name
MIN_PTS_FOR_OBB = 4                  # minimum points in an instance to create an OBB

# Class list behavior:
# - If CLASSES is non-empty: fixed mapping; unknown classes can be skipped (if FILTER_BY_CLASSES=True)
# - If CLASSES empty: auto-discover classes from labels
FILTER_BY_CLASSES = False
CLASSES: List[str] = [
    # Example:
    # "piste_left_end",
    # "piste_left_warn",
]

# Debug visualization
WRITE_DEBUG_VIZ = True
DEBUG_DIRNAME = "debug_viz"
DEBUG_THICKNESS = 2

# ============================================================
# HELPERS
# ============================================================
def canonical_class_name(label: str) -> str:
    s = label.strip()
    for suf in STRIP_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def find_image_for_json(jp: Path, data: dict, images_root: Optional[Path]) -> Optional[Path]:
    """
    Locate the image corresponding to a Labelme JSON.
    Priority:
      1) images_root / imagePath (if provided)
      2) json_parent / imagePath
      3) images_root / (json_stem + ext)
      4) json_parent / (json_stem + ext)
    """
    ip = data.get("imagePath")

    candidates: List[Path] = []
    if ip:
        if images_root is not None:
            candidates.append(images_root / Path(ip).name)
            candidates.append(images_root / ip)
        candidates.append(jp.parent / Path(ip).name)
        candidates.append(jp.parent / ip)

    for ext in IMG_EXTS:
        candidates.append(jp.with_suffix(ext))
        if images_root is not None:
            candidates.append(images_root / (jp.stem + ext))

    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def order_box_points_clockwise(box: np.ndarray) -> np.ndarray:
    """
    Ensure consistent corner order (clockwise), starting TL-ish.
    """
    pts = box.astype(np.float64)

    idx = np.lexsort((pts[:, 0], pts[:, 1]))  # sort by y then x
    pts = pts[idx]

    top = pts[:2]
    bot = pts[2:]

    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]

    ordered = np.array([tl, tr, br, bl], dtype=np.float64)

    def polygon_area(p):
        x = p[:, 0]
        y = p[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

    # make sure clockwise
    if polygon_area(ordered) < 0:
        ordered = np.array([tl, bl, br, tr], dtype=np.float64)

    return ordered

# persentage padding does not work since the lng shape of the strip
def points_to_obb(points_xy: np.ndarray, padding_px: float = 20.0) -> np.ndarray:
    """
    Fit oriented rectangle using OpenCV minAreaRect and expand it by padding_px
    on all sides (object context padding for detection).
    """
    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect

    # Expand width and height by padding on BOTH sides
    rw_p = max(1.0, rw + 2.0 * padding_px)
    rh_p = max(1.0, rh + 2.0 * padding_px)

    rect_padded = ((cx, cy), (rw_p, rh_p), angle)
    box = cv2.boxPoints(rect_padded).astype(np.float64)  # (4,2)

    return order_box_points_clockwise(box)

def norm_xy(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    out = xy.astype(np.float64).copy()
    out[:, 0] /= float(w)
    out[:, 1] /= float(h)
    return out


def write_data_yaml(out_root: Path, class_names: List[str]) -> None:
    yaml_text = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names:\n"
    )
    for i, n in enumerate(class_names):
        yaml_text += f"  {i}: {n}\n"
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def draw_debug(img_bgr: np.ndarray, instances: List[Tuple[str, int, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    instances: list of (class_name, class_id, box_px(4,2), pts_px(N,2))
    Draw OBB and points.
    """
    out = img_bgr.copy()
    for cls_name, cls_id, box, pts in instances:
        box_i = box.astype(int)
        cv2.polylines(out, [box_i], True, (0, 255, 0), DEBUG_THICKNESS)  # green
        # label near the first corner
        x0, y0 = int(box_i[0, 0]), int(box_i[0, 1])
        cv2.putText(out, f"{cls_id}:{cls_name}", (x0 + 4, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # draw points (red)
        for p in pts:
            cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    json_root = Path(JSON_ROOT)
    images_root = Path(IMAGES_ROOT) if (USE_PREUNDISTORTED and IMAGES_ROOT) else (Path(IMAGES_ROOT) if IMAGES_ROOT else None)

    if not json_root.exists():
        raise SystemExit(f"JSON_ROOT not found: {json_root.resolve()}")

    if USE_PREUNDISTORTED:
        if images_root is None or not images_root.exists():
            raise SystemExit(
                f"USE_PREUNDISTORTED=True but IMAGES_ROOT missing or not found: {IMAGES_ROOT}\n"
                f"Set IMAGES_ROOT correctly, or set USE_PREUNDISTORTED=False."
            )

    out_root = Path(OUT_ROOT)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    debug_root = out_root / DEBUG_DIRNAME
    if WRITE_DEBUG_VIZ:
        (debug_root / "train").mkdir(parents=True, exist_ok=True)
        (debug_root / "val").mkdir(parents=True, exist_ok=True)

    # Gather jsons
    json_files = sorted(json_root.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No Labelme json files found under: {json_root.resolve()}")

    # Split
    rnd = random.Random(RANDOM_SEED)
    idxs = list(range(len(json_files)))
    rnd.shuffle(idxs)

    n_val = int(round(len(idxs) * VAL_RATIO))
    val_set = set(idxs[:n_val])

    # Class mapping
    class_to_id: Dict[str, int] = {n: i for i, n in enumerate(CLASSES)} if CLASSES else {}

    processed = 0
    instances_written = 0
    instances_skipped = 0
    images_missing = 0

    for i, jp in enumerate(json_files):
        split = "val" if i in val_set else "train"

        data = json.loads(jp.read_text(encoding="utf-8"))
        img_path = find_image_for_json(jp, data, images_root)
        if img_path is None:
            images_missing += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            images_missing += 1
            continue
        h, w = img.shape[:2]

        # group points by group_id
        inst_pts: Dict[int, List[Tuple[float, float, str]]] = {}
        for sh in data.get("shapes", []):
            if sh.get("shape_type") != "point":
                continue
            label = sh.get("label", "")
            gid = sh.get("group_id", 0)
            x, y = sh["points"][0]
            inst_pts.setdefault(gid, []).append((float(x), float(y), label))

        # Prepare YOLO label lines
        yolo_lines: List[str] = []
        debug_instances: List[Tuple[str, int, np.ndarray, np.ndarray]] = []

        for gid, pts in inst_pts.items():
            if len(pts) < MIN_PTS_FOR_OBB:
                instances_skipped += 1
                continue

            cls_name = canonical_class_name(pts[0][2])

            if FILTER_BY_CLASSES and cls_name not in class_to_id:
                instances_skipped += 1
                continue

            if cls_name not in class_to_id:
                class_to_id[cls_name] = len(class_to_id)

            cls_id = class_to_id[cls_name]

            pts_xy = np.array([(x, y) for x, y, _ in pts], dtype=np.float64)
            box_px = points_to_obb(pts_xy)           # (4,2) pixels
            box_n = norm_xy(box_px, w, h)            # normalized

            box_n[:, 0] = np.clip(box_n[:, 0], 0.0, 1.0)
            box_n[:, 1] = np.clip(box_n[:, 1], 0.0, 1.0)

            flat = box_n.reshape(-1)
            yolo_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in flat.tolist()))
            instances_written += 1

            if WRITE_DEBUG_VIZ:
                debug_instances.append((cls_name, cls_id, box_px, pts_xy))

        # Copy image into dataset
        out_img = out_root / "images" / split / img_path.name
        if out_img.resolve() != img_path.resolve():
            shutil.copy2(img_path, out_img)

        # Write labels (even if empty)
        out_lbl = out_root / "labels" / split / f"{jp.stem}.txt"
        out_lbl.write_text(("\n".join(yolo_lines) + ("\n" if yolo_lines else "")), encoding="utf-8")

        # Debug viz
        if WRITE_DEBUG_VIZ:
            dbg = draw_debug(img, debug_instances)
            dbg_path = debug_root / split / f"{jp.stem}.jpg"
            cv2.imwrite(str(dbg_path), dbg)

        processed += 1

    # Final class list
    class_names = [None] * len(class_to_id)
    for name, idx in class_to_id.items():
        class_names[idx] = name

    write_data_yaml(out_root, class_names)

    print("Done.")
    print(f"  JSON processed:          {processed}")
    print(f"  Missing/Unread images:   {images_missing}")
    print(f"  Instances written:       {instances_written}")
    print(f"  Instances skipped:       {instances_skipped}")
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Output dataset:          {out_root.resolve()}")
    print(f"  data.yaml:               {(out_root / 'data.yaml').resolve()}")
    if WRITE_DEBUG_VIZ:
        print(f"  Debug viz:               {debug_root.resolve()}")


if __name__ == "__main__":
    main()
