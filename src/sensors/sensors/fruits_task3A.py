#!/usr/bin/env python3
"""
hsv_tuner.py

Default: loads ~/Downloads/hsv.jpeg
Features:
 - Trackbars: LH,LS,LV,UH,US,UV
 - Left-click to print HSV and BGR at pixel
 - 's' prints LOWER/UPPER arrays
 - 'w' writes mask and masked image to disk
 - 'q' quits
"""
import os
import sys
import argparse
import cv2
import numpy as np

DEFAULT_IMAGE = os.path.expanduser('~/Downloads/hsv.jpeg')

def nothing(x): pass

class HSVTuner:
    def __init__(self, image_path):
        self.win = 'HSV Tuner'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        # initial trackbar values (same style as your original)
        cv2.createTrackbar('LH', self.win, 0, 179, nothing)
        cv2.createTrackbar('LS', self.win, 0, 255, nothing)
        cv2.createTrackbar('LV', self.win, 0, 255, nothing)
        cv2.createTrackbar('UH', self.win, 60, 179, nothing)
        cv2.createTrackbar('US', self.win, 120, 255, nothing)
        cv2.createTrackbar('UV', self.win, 200, 255, nothing)

        self.img_path = image_path
        self.frame = None
        self.hsv = None
        self.load_image(self.img_path)

        cv2.setMouseCallback(self.win, self.on_mouse)

    def load_image(self, path):
        if not path or not os.path.exists(path):
            print(f"ERROR: image not found: {path}", file=sys.stderr)
            sys.exit(1)
        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: cv2 failed to read image: {path}", file=sys.stderr)
            sys.exit(1)
        self.frame = img
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        print(f"Loaded image: {path} (shape={self.frame.shape})")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame is None:
                return
            h, w = self.frame.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return
            px_hsv = self.hsv[y, x].tolist()
            px_bgr = self.frame[y, x].tolist()
            print(f"Clicked (x={x}, y={y}) -> HSV={px_hsv}, BGR={px_bgr}")

    def get_trackbar_hsv(self):
        lh = cv2.getTrackbarPos('LH', self.win)
        ls = cv2.getTrackbarPos('LS', self.win)
        lv = cv2.getTrackbarPos('LV', self.win)
        uh = cv2.getTrackbarPos('UH', self.win)
        us = cv2.getTrackbarPos('US', self.win)
        uv = cv2.getTrackbarPos('UV', self.win)
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        return lower, upper

    def run(self):
        print("Running HSV Tuner. Click to sample pixel HSV. 's' prints ranges. 'w' writes files. 'q' quits.")
        while True:
            img = self.frame.copy()
            self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower, upper = self.get_trackbar_hsv()

            # handle hue wrap-around (e.g., LH > UH when selecting red)
            if lower[0] <= upper[0]:
                mask = cv2.inRange(self.hsv, lower, upper)
            else:
                # split into two ranges [LH,179] and [0,UH]
                lower1 = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
                upper1 = np.array([179, upper[1], upper[2]], dtype=np.uint8)
                lower2 = np.array([0, lower[1], lower[2]], dtype=np.uint8)
                upper2 = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
                mask1 = cv2.inRange(self.hsv, lower1, upper1)
                mask2 = cv2.inRange(self.hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)

            masked_color = cv2.bitwise_and(img, img, mask=mask)
            overlay = cv2.addWeighted(img, 0.6, masked_color, 0.4, 0)

            # show current ranges on overlay
            cv2.putText(overlay, f"LOW={list(lower)} UPP={list(upper)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(self.win, overlay)
            k = cv2.waitKey(30) & 0xFF

            if k == ord('q'):
                print("Quitting.")
                break
            if k == ord('s'):
                print(f"LOWER={list(lower)}, UPPER={list(upper)}")
            if k == ord('w'):
                base = os.path.splitext(os.path.basename(self.img_path))[0]
                out_mask = f"{base}_mask.png"
                out_masked = f"{base}_masked.png"
                cv2.imwrite(out_mask, mask)
                cv2.imwrite(out_masked, masked_color)
                print(f"Wrote: {out_mask}, {out_masked}")

        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', '-i', type=str, default=DEFAULT_IMAGE,
                   help="Path to image. Default: ~/Downloads/hsv.jpeg")
    return p.parse_args()

def main():
    args = parse_args()
    tuner = HSVTuner(args.image)
    tuner.run()

if __name__ == '__main__':
    main()
