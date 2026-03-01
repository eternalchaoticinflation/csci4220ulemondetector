# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:14:25 2026

@author: Wei Cui
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
plan, we will crop manually and canny will judge how good a crop you have.
once it is acceptable above a threshold, it will print message good template!"
"""



img = cv2.imread("template2.jpg")
H, W = img.shape[:2]

# make a preview ~900px wide
scale = 900 / W
small = cv2.resize(img, (int(W*scale), int(H*scale)))

cv2.namedWindow("Crop template (preview)", cv2.WINDOW_NORMAL)
roi_s = cv2.selectROI("Crop template (preview)", small, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

xs, ys, ws, hs = map(int, roi_s)

# map ROI back to original coordinates
x = int(xs / scale); y = int(ys / scale)
w = int(ws / scale); h = int(hs / scale)

crop = img[y:y+h, x:x+w].copy()
#force crop save to be small

max_dim = 300  # pick 200-400
h, w = crop.shape[:2]
s = max_dim / max(h, w)
if s < 1:
    crop_small = cv2.resize(crop, (int(w*s), int(h*s)))
    cv2.imwrite("template1_manualcrop_small.png", crop_small)
    print("saved template1_manualcrop_small.png", crop_small.shape)
    
def canny_crop_score(crop_bgr, low=40, high=120, border=8):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)

    edges = cv2.Canny(g, low, high)

    # edge density
    ed = edges.mean() / 255.0  # fraction of pixels that are edges

    # border edge fraction (penalize cut-off object or tight crop)
    h, w = edges.shape
    b = border
    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True

    if edges.sum() == 0:
        return {"edge_density": ed, "border_frac": 1.0, "score": 0.0}

    border_frac = edges[border_mask].sum() / edges.sum()

    # heuristic target range for edge density:
    # too low => blank; too high => noisy background
    # you can tune these after 1-2 tests
    ed_ok = np.exp(-((ed - 0.08) / 0.06)**2)   # peak around 0.08
    bf_ok = np.exp(-((border_frac - 0.08) / 0.08)**2)  # peak around 0.08

    score = float(ed_ok * bf_ok)

    return {"edge_density": float(ed), "border_frac": float(border_frac), "score": score}

"""
you crop, Gaussian get rid of the noise, together give you a score
to see if you pass of FFT!?
"""
def show_edges(crop_bgr, low=40, high=120):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    edges = cv2.Canny(g, low, high)

    plt.figure()
    plt.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Crop")
    plt.axis("off")

    plt.figure()
    plt.imshow(edges, cmap="gray")
    plt.title("Canny edges")
    plt.axis("off")
    plt.show()
    
crop = cv2.imread("template1_manualcrop_small.png")
assert crop is not None

show_edges(crop, low=40, high=120)          # visual check
score = canny_crop_score(crop, 40, 120, 8)  # numeric check
print(score)
#print good template or else
ok = score["score"] > 0.25 and score["border_frac"] < 0.20
print("GOOD TEMPLATE" if ok else "re-crop tighter/cleaner")