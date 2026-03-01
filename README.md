# csci4220ulemondetector
submission for lab3
LEMON DETECTOR (FFT CROSS-CORRELATION) — HOW TO USE

Overview
--------
This script detects and counts lemons using Fourier-domain cross-correlation (FFT).
It supports:
- 2 templates (two lemon poses)
- multi-scale pyramid
- rotation search
- a yellow HSV mask to reduce false positives

Important: I cannot submit pictures in the repo.
So the TA / user must download the target and template images from the lab materials
and place them locally.
you can get the pictures here [https://csundergrad.science.uoit.ca/courses/csci4220u/latest/labs/04-fourier-space-object-detection.html]

I have also uploaded the files with templates and targets on my github. 

Requirements
------------
- Python 3.9+ (3.10/3.11 recommended)
- OpenCV + NumPy

Install:
  pip install opencv-python numpy

Files
-----
- lemondetector.py  (main detector)
- template maker tool called makeTemplate.py
- images (they MAY NOT be included in the submission but you can go to my github repo):
    target images (cropped/small)
    template1_manualcrop_small.png
    template2_manualcrop_small.png

CRITICAL: Image size (do NOT use huge originals)
-----------------------------------------------
The detector is slow and unreliable on very large images.

Target images must be ~950x950 (roughly).
Do NOT use the default full-size images if they are huge.

For each target image:
- screenshot / snip it to around 950x950
- save it as something like:
    target1small.png
    target2small.png
    target3small.png

Step-by-step usage
------------------
1) Download the lab images locally
   - target images (lemon grids)
   - template images (lemons)
   Put them somewhere on your computer.

2) Make GOOD templates (VERY IMPORTANT)
   You need TWO templates:
   - Template 1: lemon pose A
   - Template 2: lemon pose B

   Use the Template Maker until it reports:
     "GOOD TEMPLATE"

   Save the templates as SMALL CROPS (a few hundred pixels wide/tall).
   Example filenames (recommended):
     template1_manualcrop_small.png
     template2_manualcrop_small.png

   Put these two template files in the SAME folder as lemondetector.py
   (or remember where they are if using the file picker).

   WARNING:
   Do NOT use a full, uncropped template image.
   A huge template causes huge boxes and bad matches.

3) Run the detector
   Open a terminal in the folder containing lemondetector.py and run:
     python lemondetector.py

   The script will:
   - ask you to pick the TARGET image
   - ask you to pick TEMPLATE 1
   - ask you to pick TEMPLATE 2
   - run detection and print the count
   - save outputs next to the target image:
       *_detected.png  (green dots at detected lemon centers)
       *_mask.png      (yellow mask debug)

Output interpretation
---------------------
- The printed count is the number of fully matchable lemons (full lemons).
- Lemons cut off by the image border may be missed because the algorithm uses
  VALID correlation (template must fit fully inside the image).

Knobs (tuning)
--------------
Inside lemondetector.py you can adjust:
- thr_rel (threshold):
    higher -> fewer detections / fewer false positives
    lower  -> more detections / more false positives
- scales:
    widen if lemon size varies
    narrow if lemon size is consistent
- angles:
    smaller step detects more rotations but runs slower

Suggested starting settings:
  thr_rel = 0.60
  scales = [0.8, 0.9, 1.0, 1.1]
  angles = 0..360 step 15 degrees

If too many false positives:
  increase thr_rel (example: 0.60 -> 0.65)

If missing lemons:
  decrease thr_rel (example: 0.60 -> 0.55)
  widen scales (example: add 0.75 or 1.15)
  use finer angles (15 deg instead of 30 deg)

Notes / limitations
-------------------
- Template matching expects the object to be present in full.
  Partially visible lemons (cut off by the border) may not be detected.
- The yellow mask assumes lemons are yellow-ish.
  If lighting is unusual, HSV thresholds may need loosening in yellow_mask().
