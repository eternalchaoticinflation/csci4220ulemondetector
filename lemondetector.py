# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# ---------- "press any key" helper ----------
def wait_prompt(msg="Continue? [y] "):
    while True:
        s = input(msg).strip().lower()
        if s in ("", "y", "yes"):
            return
        if s in ("n", "no", "q", "quit"):
            raise SystemExit("User cancelled.")
        print("Type y (or Enter) to continue, n to quit.")

# ---------- file picker ----------
def pick_file(title, filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*"))):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

# ----------------------------
# Yellow mask (HSV)
# ----------------------------
def yellow_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = (15, 40, 40)
    upper = (45, 255, 255)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m

# ----------------------------
# Rotate template (expand canvas)
# ----------------------------
def rotate_expand(img_bgr, angle_deg):
    h, w = img_bgr.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(
        img_bgr, M, (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

# ----------------------------
# Feature preprocessing
# (mask-safe: normalize only inside mask)
# ----------------------------
def preprocess_feature(img_bgr, use_mask=False):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    if use_mask:
        m = (yellow_mask(img_bgr) > 0)
        out = np.zeros_like(mag, dtype=np.float32)
        if np.any(m):
            vals = mag[m]
            mu = float(vals.mean())
            out[m] = vals - mu
            denom = float(np.linalg.norm(out[m]) + 1e-8)
            out[m] /= denom
        return out

    mag -= float(mag.mean())
    mag /= float(np.linalg.norm(mag) + 1e-8)
    return mag

# ----------------------------
# FFT valid cross-correlation
# ----------------------------
def fft_corr_valid(I, T):
    H, W = I.shape
    h, w = T.shape
    P, Q = H + h - 1, W + w - 1
    FI = np.fft.fft2(I, s=(P, Q))
    FT = np.fft.fft2(T, s=(P, Q))
    C = np.fft.ifft2(FI * np.conj(FT)).real
    return C[h - 1:H, w - 1:W]

# ----------------------------
# Peak picking (local maxima + NMS)
# ----------------------------
def find_peaks_nms(C, min_dist=40, thr_rel=0.60):
    Cn = C.astype(np.float32)
    Cn -= float(Cn.min())
    mx = float(Cn.max())
    if mx < 1e-12:
        return []
    Cn /= mx

    k = 2 * min_dist + 1
    dil = cv2.dilate(Cn, np.ones((k, k), np.uint8))
    ys, xs = np.where((Cn >= dil - 1e-6) & (Cn >= thr_rel))

    pts = [(int(x), int(y), float(Cn[y, x])) for x, y in zip(xs, ys)]
    pts.sort(key=lambda t: t[2], reverse=True)

    keep = []
    for x, y, s in pts:
        if all((x - x2) ** 2 + (y - y2) ** 2 >= min_dist ** 2 for x2, y2, _ in keep):
            keep.append((x, y, s))
    return keep

# ----------------------------
# Center-in-mask check (tolerant)
# ----------------------------
def center_is_yellow(mask_u8, cx, cy, r=7, frac=0.20):
    H, W = mask_u8.shape
    x0 = max(0, int(cx) - r); x1 = min(W, int(cx) + r + 1)
    y0 = max(0, int(cy) - r); y1 = min(H, int(cy) + r + 1)
    patch = (mask_u8[y0:y1, x0:x1] > 0)
    return patch.mean() >= frac

# ----------------------------
# Full detector: multi-template, multi-scale, multi-rotation
# ----------------------------
def count_fft_multi_scale_rot(image_bgr, templates_bgr, scales, angles, thr_rel=0.60):
    I = preprocess_feature(image_bgr, use_mask=True)
    m = yellow_mask(image_bgr)

    dets = []  # (cx,cy,score,w,h,x0,y0)

    for tmpl in templates_bgr:
        for s in scales:
            Ts = cv2.resize(tmpl, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if Ts.shape[0] < 30 or Ts.shape[1] < 30:
                continue

            for a in angles:
                Tr = rotate_expand(Ts, a) if a != 0 else Ts
                if Tr.shape[0] < 30 or Tr.shape[1] < 30:
                    continue

                T = preprocess_feature(Tr, use_mask=False)
                C = fft_corr_valid(I, T)

                h, w = T.shape
                min_dist = int(0.65 * min(h, w))
                peaks = find_peaks_nms(C, min_dist=min_dist, thr_rel=thr_rel)

                for x0, y0, score in peaks:
                    cx, cy = x0 + w // 2, y0 + h // 2
                    dets.append((cx, cy, score, w, h, x0, y0))

    # global NMS across all detections
    dets.sort(key=lambda t: t[2], reverse=True)

    keep = []
    for cx, cy, score, w, h, x0, y0 in dets:
        # reject if center not yellow (tolerant patch)
        if not center_is_yellow(m, cx, cy):
            continue

        md = int(0.75 * min(w, h))
        if all((cx - kcx) ** 2 + (cy - kcy) ** 2 >= min(md, kmd) ** 2
               for (kcx, kcy, ks, kw, kh, kx, ky, kmd) in keep):
            keep.append((cx, cy, score, w, h, x0, y0, md))

    boxes = [(x0, y0, x0 + w, y0 + h) for (cx, cy, score, w, h, x0, y0, md) in keep]
    return len(boxes), boxes, keep

# ----------------------------
# MAIN (interactive)
# ----------------------------
if __name__ == "__main__":
    print("\n=== FFT Lemon Counter ===\n")

   # wait_any_key("Press any key to pick the TARGET image...")
    wait_prompt("Pick TARGET image now? [y] ")
    target_path = pick_file("Select TARGET image")
    if not target_path:
        raise SystemExit("No target selected. Exiting.")

    #wait_any_key("Press any key to pick TEMPLATE 1 (good crop from template maker)...")
    wait_prompt("Pick TEMPLATE 1 now? [y] ")
    t1_path = pick_file("Select TEMPLATE 1")
    if not t1_path:
        raise SystemExit("No template 1 selected. Exiting.")

    
#wait_any_key("Press any key to pick TEMPLATE 2 (second pose)...")
    wait_prompt("Pick TEMPLATE 2 now? [y] ")
    t2_path = pick_file("Select TEMPLATE 2")
    if not t2_path:
        raise SystemExit("No template 2 selected. Exiting.")

    img = cv2.imread(target_path);  assert img is not None, "Failed to load target"
    t1  = cv2.imread(t1_path);      assert t1  is not None, "Failed to load template 1"
    t2  = cv2.imread(t2_path);      assert t2  is not None, "Failed to load template 2"

    # knobs (start here)
    scales = [0.8, 0.9, 1.0, 1.1]
    angles = list(range(0, 360, 15))
    thr_rel = 0.60

    count, boxes, keep = count_fft_multi_scale_rot(img, [t1, t2], scales, angles, thr_rel=thr_rel)
    print("\nDetected full lemons =", count)

    out = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 8, (0, 255, 0), -1)

    base, ext = os.path.splitext(target_path)
    out_path = base + "_detected.png"
    mask_path = base + "_mask.png"

    cv2.imwrite(out_path, out)
    cv2.imwrite(mask_path, yellow_mask(img))
    print("Wrote:", out_path)
    print("Wrote:", mask_path)

    print("Done.")
    print("\n--- Tuning tips ---")
print(f"Current knobs: thr_rel={thr_rel}, scales={scales}, angle_step={angles[1]-angles[0] if len(angles)>1 else 0}")
print("If you get TOO MANY false positives:")
print("  - increase thr_rel (e.g. 0.60 -> 0.65)")
print("  - or make angles coarser (15° -> 30°) to reduce random matches")
print("  - or tighten scales range (remove 0.8/1.1 if size is stable)")
print("If you MISS lemons (false negatives):")
print("  - decrease thr_rel (e.g. 0.60 -> 0.55)")
print("  - or make angles finer (30° -> 15°)")
print("  - or widen scales (add 0.75 or 1.15)")
print("Note: border-cut lemons usually won't be detected with 'valid' correlation; use the mask (connected components) to count partials.")