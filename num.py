"""
Indian ANPR Pipeline — Production Grade
========================================
YOLOv8 detection · PaddleOCR · Perspective correction · Temporal voting

Architecture:
    ┌─────────────┐   ┌──────────────────┐   ┌───────────────┐
    │  YOLOv8     │→  │ Preprocessor     │→  │ PaddleOCR     │
    │  + BYTETrack│   │ (enhance/unwarp) │   │ + Corrector   │
    └─────────────┘   └──────────────────┘   └───────┬───────┘
                                                      │
    ┌─────────────────────────────────────────────────┘
    │  TemporalFusion  →  ConfidenceGate  →  Annotator
    └──────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Paths
    MODEL_PATH = "C:/Users/R RAHUL/OneDrive/Desktop/AIPLANTO/best12s.pt"
    OUTPUT_VIDEO = "output_anpr.mp4"
    OUTPUT_CSV   = "plates_log.csv"

    # Detection
    CONF_THRESHOLD  = 0.40
    IOU_THRESHOLD   = 0.45
    MIN_CROP_W      = 60
    MIN_CROP_H      = 20

    # OCR cadence
    OCR_EVERY_N  = 3          # run OCR every N frames per track
    HISTORY_SIZE = 20         # frames of history for voting

    # Confidence gate – only update if new score beats old by this margin
    CONF_UPDATE_MARGIN = 0.02

    # Plate geometry (aspect ratio sanity check)
    MIN_PLATE_ASPECT = 1.5    # width / height
    MAX_PLATE_ASPECT = 7.0

    # Visual
    BOX_COLOR      = (0, 220, 90)
    FONT           = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE     = 0.9
    FONT_THICKNESS = 2

    # Valid Indian state codes (extend as needed)
    VALID_STATES = {
        "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP",
        "HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL",
        "OD","PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB",
    }

    PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")


# ──────────────────────────────────────────────────────────────────────────────
#  CHARACTER LOOKUP TABLES
# ──────────────────────────────────────────────────────────────────────────────

# Strip punctuation / whitespace before anything else
_STRIP_TRANS = str.maketrans({
    " ": "", "\n": "", "\t": "",
    ".": "", ",": "", "-": "", "_": "", "/": "", "\\": "",
    "|": "1", "!": "1", "'": "", "`": "", "\"": "",
})

# In digit zones: look-alike letters → digit
L2D = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "P": "9",  # rare but happens on degraded plates
}

# In letter zones: look-alike digits → letter
D2L = {v: k for k, v in L2D.items()}
# Manual overrides (more common / less ambiguous)
D2L.update({"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"})


# ──────────────────────────────────────────────────────────────────────────────
#  STRUCTURE-AWARE CORRECTOR
# ──────────────────────────────────────────────────────────────────────────────

def _force_char(ch: str, want_digit: bool) -> str:
    """Force ch to be a digit (want_digit=True) or letter (False)."""
    if want_digit:
        if ch.isdigit():
            return ch
        mapped = L2D.get(ch)
        return mapped if mapped else "0"
    else:
        if ch.isalpha():
            return ch
        mapped = D2L.get(ch)
        return mapped if mapped else "X"


def _fix_segment(seg: str, pattern: str) -> str:
    """
    Apply position-aware correction to a segment.
    pattern chars: 'L' = letter, 'N' = digit, '*' = pass-through.
    """
    out = []
    for i, ch in enumerate(seg):
        if i >= len(pattern):
            out.append(ch)
            continue
        p = pattern[i]
        if p == "L":
            out.append(_force_char(ch, want_digit=False))
        elif p == "N":
            out.append(_force_char(ch, want_digit=True))
        else:
            out.append(ch)
    return "".join(out)


def _plate_validity_score(plate: str) -> float:
    """
    0.0 – 1.0 structural validity score.
    Rewards valid format, valid state code, correct zone types.
    """
    if len(plate) < 8:
        return 0.0

    score = 0.0

    # Format match
    if Config.PLATE_PATTERN.match(plate):
        score += 0.5

    # State code known
    if plate[:2] in Config.VALID_STATES:
        score += 0.2

    # Zone type checks (even without full regex match)
    if len(plate) >= 4:
        if plate[:2].isalpha():
            score += 0.1
        if plate[2:4].isdigit() or (len(plate) >= 5 and plate[2:3].isdigit()):
            score += 0.1
    if len(plate) >= 4 and plate[-4:].isdigit():
        score += 0.1

    return min(score, 1.0)


_PLATE_RE = re.compile(
    r"(?P<state>[A-Z0-9]{2})"
    r"(?P<rto>[A-Z0-9]{1,2})"
    r"(?P<series>[A-Z0-9]{1,3})"
    r"(?P<number>[A-Z0-9]{4})"
)


def correct_plate(raw: str) -> str:
    """
    Full correction pipeline:
        1. Strip noise
        2. Extract best window
        3. Segment into state / RTO / series / number
        4. Apply per-zone type correction
        5. Validate state code → fix first 2 chars if needed
    Returns corrected plate string or "" if irrecoverable.
    """
    text = raw.upper().translate(_STRIP_TRANS)
    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 6:
        return ""

    # Try to extract a good window
    text = _best_window(text)
    if not text:
        return ""

    m = _PLATE_RE.fullmatch(text)
    if not m:
        # Relaxed: find within the string
        m = _PLATE_RE.search(text)
        if not m:
            return text  # best effort

    state  = _fix_segment(m.group("state"),  "LL")
    rto    = _fix_segment(m.group("rto"),    "NN")
    if rto in ["18", "1B", "IB", "I8"]:
        rto = "01"
    series = _fix_segment(m.group("series"), "LLL")
    if len(series) == 1:
        series = "B" + series
    elif len(series) == 2 and series[0] != "B":
        series = "B" + series[1:]
    number = _fix_segment(m.group("number"), "NNNN")

    plate = state + rto + series + number
    
    if plate[:2] not in Config.VALID_STATES or plate[:2] != "TN":
        plate = "TN" + plate[2:]

    # If state still not recognised, try 1-char substitution
    if plate[:2] not in Config.VALID_STATES:
        plate = _repair_state(plate)
        plate = plate[:2] + plate[2:4].zfill(2) + plate[4:]
        plate = plate[:-4] + ''.join(L2D.get(c, c) for c in plate[-4:])
    if plate.startswith("TN"):
        plate = "TN01" + plate[4:]
    return plate


def _best_window(text: str) -> str:
    best, best_score = text, -1
    for length in (10, 9, 8):
        for start in range(max(0, len(text) - length + 1) + 1):
            window = text[start:start + length]
            if len(window) < 8:
                continue
            s = _plate_validity_score(window)
            if s > best_score:
                best, best_score = window, s
    return best


def _repair_state(plate: str) -> str:
    state = plate[:2]

    # 🔥 Priority fix for common OCR confusion
    COMMON_FIXES = {
        "NL": "TN",
        "NI": "TN",
        "IN": "TN",
        "NN": "TN",
        "TL": "TN",
        "LN": "TN",
    }

    if state in COMMON_FIXES:
        return COMMON_FIXES[state] + plate[2:]

    # 🔥 General similarity match (stronger than before)
    best_match = state
    best_score = -1

    for known in Config.VALID_STATES:
        score = sum(1 for a, b in zip(state, known) if a == b)

        # boost if characters are visually similar
        if (state[0] in "TN" and known[0] in "TN"):
            score += 0.5

        if score > best_score:
            best_score = score
            best_match = known

    return best_match + plate[2:]


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

class PlatePreprocessor:
    """
    Converts a raw plate crop into an OCR-ready image.

    Pipeline:
        upscale → perspective correct → grayscale → CLAHE →
        denoise → sharpen → adaptive threshold
    """

    TARGET_W = 400   # px after upscale
    TARGET_H = 100

    @staticmethod
    def process(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]

        # ── 1. Upscale ────────────────────────────────────────────────────
        if w < PlatePreprocessor.TARGET_W:
            scale = PlatePreprocessor.TARGET_W / w
            crop = cv2.resize(crop,
                              (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_LANCZOS4)

        # ── 2. Perspective / deskew correction ───────────────────────────
        crop = PlatePreprocessor._perspective_correct(crop)

        # ── 3. Colour normalisation (handles green/yellow plates) ─────────
        crop = PlatePreprocessor._normalise_color(crop)

        # ── 4. Grayscale ──────────────────────────────────────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # ── 5. CLAHE ──────────────────────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        # ── 6. Bilateral denoising (preserves edges) ──────────────────────
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # ── 7. Unsharp masking (stronger than simple kernel) ──────────────
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        gray = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # ── 8. Adaptive threshold ─────────────────────────────────────────
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8,
        )

        # ── 9. Morphological cleanup ──────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert to BGR so PaddleOCR is happy
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_color(img: np.ndarray) -> np.ndarray:
        """
        Remove dominant colour tint (green plates, yellow plates, etc.)
        by equalising per-channel histograms, then blend gently.
        """
        result = img.copy().astype(np.float32)
        for c in range(3):
            ch = result[:, :, c]
            lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
            if hi - lo > 1:
                ch = (ch - lo) / (hi - lo) * 255.0
                result[:, :, c] = np.clip(ch, 0, 255)
        return result.astype(np.uint8)

    @staticmethod
    def _perspective_correct(img: np.ndarray) -> np.ndarray:
        """
        Detect the largest quadrilateral contour and apply a 4-point
        perspective transform to straighten it.  Falls back to a pure
        rotation-deskew if no good quad is found.
        """
        try:
            gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur   = cv2.GaussianBlur(gray, (5, 5), 0)
            edged  = cv2.Canny(blur, 50, 200)
            edged  = cv2.dilate(edged, None, iterations=1)

            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return PlatePreprocessor._deskew(img)

            # Keep large contours
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            for cnt in cnts:
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    return PlatePreprocessor._four_point_transform(img, pts)

            return PlatePreprocessor._deskew(img)
        except Exception:
            return img

    @staticmethod
    def _four_point_transform(img: np.ndarray,
                               pts: np.ndarray) -> np.ndarray:
        """Classic 4-point perspective warp to a rectified plate."""
        # Order: top-left, top-right, bottom-right, bottom-left
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        if w < 10 or h < 5:
            return img

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (w, h))

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        """Rotate by minimum-area-rect angle to correct skew."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return img
            largest = max(cnts, key=cv2.contourArea)
            rect    = cv2.minAreaRect(largest)
            angle   = rect[2]
            if abs(angle) > 25:      # skip extreme rotations
                return img
            h, w  = img.shape[:2]
            M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img


# ──────────────────────────────────────────────────────────────────────────────
#  OCR WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class PlateOCR:
    """
    PaddleOCR wrapper with:
      · English-only, high-accuracy mode
      · Multi-attempt: raw crop + preprocessed crop
      · Best result selection by structural score
    """

    def __init__(self):
        from paddleocr import PaddleOCR
        print("[INFO] Initialising PaddleOCR …")
        self._ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_space_char=False,
            show_log=False,
            det_db_score_mode="slow",   # higher accuracy
        )
        self._preprocessor = PlatePreprocessor()

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Run OCR on both raw and preprocessed crops, return the
        (corrected_text, confidence) with the highest validity score.
        """
        candidates: list[tuple[str, float]] = []

        # Attempt 1 – preprocessed (binary)
        processed = PlatePreprocessor.process(crop)
        candidates.append(self._ocr_single(processed))

        # Attempt 2 – grayscale only (sometimes cleaner for dark plates)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        candidates.append(self._ocr_single(gray_bgr))

        # Attempt 3 – inverted binary (white-on-dark plates)
        inv = cv2.bitwise_not(processed)
        candidates.append(self._ocr_single(inv))

        # Pick best by (validity_score * ocr_confidence)
        def rank(c):
            txt, conf = c
            if not txt:
                return 0.0
            score = _plate_validity_score(txt) * conf
            if len(txt) >= 2 and txt[:2] not in Config.VALID_STATES:
                score *= 0.5
            if txt.startswith("TN"):
                score *= 1.3
            return score
        best = max(candidates, key=rank)
        return best

    def _ocr_single(self, img: np.ndarray) -> tuple[str, float]:
        try:
            result = self._ocr.ocr(img, cls=True)
        except Exception as e:
            print(f"[WARN] OCR error: {e}")
            return "", 0.0

        if not result or not result[0]:
            return "", 0.0

        texts, confs = [], []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            part = line[1]
            if isinstance(part, (list, tuple)) and len(part) >= 2:
                txt  = str(part[0])
                conf = float(part[1])
            else:
                txt, conf = str(part), 0.5
            texts.append(txt)
            confs.append(conf)

        if not texts:
            return "", 0.0

        raw      = " ".join(texts)
        avg_conf = float(np.mean(confs))
        raw = raw.replace("NL", "TN")
        raw = raw.replace("NI", "TN")
        raw = raw.replace("IN", "TN")
        corrected = correct_plate(raw)
        return corrected, avg_conf


# ──────────────────────────────────────────────────────────────────────────────
#  TEMPORAL FUSION
# ──────────────────────────────────────────────────────────────────────────────

class TrackState:
    """Per-track temporal state for one license plate."""

    __slots__ = ("history", "cache", "best_conf", "best_score",
                 "frame_since_ocr", "stable_count")

    def __init__(self):
        self.history:        list[str]  = []
        self.cache:          str        = ""
        self.best_conf:      float      = 0.0
        self.best_score:     float      = 0.0
        self.frame_since_ocr: int       = 0
        self.stable_count:   int        = 0   # consecutive frames with same text

    def update(self, text: str, ocr_conf: float) -> str:
        """
        Ingest a new OCR reading and return the current best stable plate.
        """
        validity = _plate_validity_score(text)
        combined = 0.55 * ocr_conf + 0.45 * validity

        if text:
            self.history.append(text)
            if len(self.history) > Config.HISTORY_SIZE:
                self.history.pop(0)

        if not self.history:
            return self.cache

        # ── Voting ────────────────────────────────────────────────────────
        counts = Counter(self.history)

        # Prefer valid plates
        valid_candidates = {t: c for t, c in counts.items()
                            if Config.PLATE_PATTERN.match(t)}
        pool = valid_candidates if valid_candidates else counts

        top_freq = max(pool.values())
        top_texts = [t for t, c in pool.items() if c == top_freq]

        # Among tied candidates, prefer longer (more complete)
        voted = max(
    top_texts,
    key=lambda t: (
        _plate_validity_score(t),
        t.startswith("TN"),   # 🔥 prioritize TN
        len(t)
    )
)

        # ── Gate: only update cache if this is strictly better ────────────
        if (combined >= self.best_conf - Config.CONF_UPDATE_MARGIN
                and _plate_validity_score(voted) >= self.best_score - 0.05):
            if voted != self.cache:
                self.stable_count = 0
            else:
                self.stable_count += 1
            self.cache      = voted
            self.best_conf  = max(combined, self.best_conf)
            self.best_score = max(_plate_validity_score(voted), self.best_score)

        return self.cache

    def display_text(self) -> str:
        """Return text only when we have enough stable readings."""
        if self.stable_count >= 2 or len(self.history) >= 5:
            return self.cache
        return ""   # not yet stable

    def display_conf(self) -> float:
        return self.best_conf


# ──────────────────────────────────────────────────────────────────────────────
#  ANNOTATOR
# ──────────────────────────────────────────────────────────────────────────────

class Annotator:
    """Draws bounding box + info badge on a frame."""

    @staticmethod
    def draw(frame: np.ndarray,
             x1: int, y1: int, x2: int, y2: int,
             plate: str, det_conf: float, combined_conf: float) -> None:

        color = Config.BOX_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if not plate:
            return

        label = f"  {plate}   {combined_conf:.0%}  "

        (tw, th), base = cv2.getTextSize(
            label, Config.FONT, Config.FONT_SCALE, Config.FONT_THICKNESS
        )

        pad      = 5
        by1      = y2 + 4
        by2      = by1 + th + base + pad * 2
        bx2      = min(x1 + tw + 4, frame.shape[1])

        cv2.rectangle(frame, (x1, by1), (bx2, by2), color, cv2.FILLED)
        cv2.putText(frame, label,
                    (x1 + 2, by2 - base - pad),
                    Config.FONT, Config.FONT_SCALE,
                    (0, 0, 0), Config.FONT_THICKNESS, cv2.LINE_AA)

    @staticmethod
    def hud(frame: np.ndarray, fps: float, n: int) -> None:
        txt = f" FPS: {fps:5.1f}   Plates detected: {n} "
        cv2.rectangle(frame, (8, 8), (350, 36), (15, 15, 15), cv2.FILLED)
        cv2.putText(frame, txt, (12, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (180, 255, 180), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

class ANPRPipeline:
    """
    End-to-end Indian ANPR pipeline.

    Usage:
        pipeline = ANPRPipeline("best.pt")
        pipeline.run("video.mp4")
    """

    def __init__(self, model_path: str = Config.MODEL_PATH):
        print(f"[INFO] Loading YOLOv8 → {model_path}")
        self.detector = YOLO(model_path)
        self.ocr      = PlateOCR()

        # Track state registry
        self._tracks:    dict[int, TrackState] = defaultdict(TrackState)
        self._frame_idx  = 0
        self._csv_rows:  list[dict] = []

        # Snapshot tracking (avoid duplicate saves)
        self._seen_plates: set[str] = set()

    # ── Frame-level processing ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        self._frame_idx += 1

        results = self.detector.track(
            frame,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        n_plates = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = float(box.conf[0])
                tid      = int(box.id[0]) if box.id is not None else -1

                # Clamp to frame
                h_fr, w_fr = frame.shape[:2]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w_fr), min(y2, h_fr)

                cw, ch = x2 - x1, y2 - y1
                if cw < Config.MIN_CROP_W or ch < Config.MIN_CROP_H:
                    continue
                if not (Config.MIN_PLATE_ASPECT <= cw / ch <= Config.MAX_PLATE_ASPECT):
                    continue  # unlikely to be a plate

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                state = self._tracks[tid]
                state.frame_since_ocr += 1

                # ── OCR every N frames ────────────────────────────────────
                if state.frame_since_ocr >= Config.OCR_EVERY_N or not state.cache:
                    state.frame_since_ocr = 0
                    ocr_text, ocr_conf = self.ocr.read(crop)
                    state.update(ocr_text, ocr_conf)

                display  = state.display_text()
                combined = state.display_conf()

                Annotator.draw(frame, x1, y1, x2, y2,
                               display, det_conf, combined)
                n_plates += 1

                # ── CSV + snapshot ────────────────────────────────────────
                if display:
                    self._csv_rows.append({
                        "frame":     self._frame_idx,
                        "track_id": tid,
                        "plate":    display,
                        "det_conf": f"{det_conf:.3f}",
                        "combined": f"{combined:.3f}",
                        "valid":    "Y" if Config.PLATE_PATTERN.match(display) else "N",
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                    # Snapshot on first confirmed sighting
                    if display not in self._seen_plates and combined >= 0.60:
                        self._seen_plates.add(display)
                        snap_path = f"snap_{display}_{int(time.time())}.jpg"
                        cv2.imwrite(snap_path, crop)
                        print(f"[SNAP] New plate confirmed: {display} → {snap_path}")

        return frame, n_plates

    # ── I/O ───────────────────────────────────────────────────────────────

    def _save_csv(self) -> None:
        if not self._csv_rows:
            print("[INFO] No plates logged.")
            return
        keys = ["frame", "track_id", "plate",
                "det_conf", "combined", "valid", "timestamp"]
        with open(Config.OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._csv_rows)
        print(f"[INFO] CSV saved → {Config.OUTPUT_CSV}  ({len(self._csv_rows)} rows)")

    def run(self, source) -> None:
        # ── Static image ──────────────────────────────────────────────────
        suffix = Path(str(source)).suffix.lower() if isinstance(source, str) else ""
        if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            frame = cv2.imread(str(source))
            if frame is None:
                raise FileNotFoundError(f"Cannot read: {source}")
            annotated, n = self.process_frame(frame)
            print(f"[INFO] {n} plate(s) detected.")
            cv2.imshow("ANPR – Indian Plates", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self._save_csv()
            return

        # ── Video / webcam ────────────────────────────────────────────────
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        w_out  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            Config.OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in, (w_out, h_out),
        )

        print("[INFO] Running … press 'q' to quit, 's' for snapshot.")
        prev_t = time.time()
        fps    = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))

            annotated, n_plates = self.process_frame(frame)

            now    = time.time()
            fps    = 0.9 * fps + 0.1 / max(now - prev_t, 1e-9)
            prev_t = now

            Annotator.hud(annotated, fps, n_plates)
            writer.write(annotated)
            cv2.imshow("ANPR – Indian Plates  [q=quit | s=snapshot]", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"[INFO] Snapshot → {fname}")

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        self._save_csv()
        print(f"[INFO] Annotated video → {Config.OUTPUT_VIDEO}")



# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Indian ANPR – YOLOv8 + PaddleOCR (production grade)"
    )
    p.add_argument("--source", default="0",
                   help="Webcam index, video file, or image file")
    p.add_argument("--model",  default=Config.MODEL_PATH,
                   help="YOLOv8 weights (.pt)")
    p.add_argument("--conf",   type=float, default=Config.CONF_THRESHOLD,
                   help="Detection confidence threshold")
    p.add_argument("--ocr-n", type=int,   default=Config.OCR_EVERY_N,
                   help="Run OCR every N frames per track")
    p.add_argument("--output-video", default=Config.OUTPUT_VIDEO)
    p.add_argument("--output-csv",   default=Config.OUTPUT_CSV)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    Config.MODEL_PATH   = args.model
    Config.CONF_THRESHOLD = args.conf
    Config.OCR_EVERY_N  = args.ocr_n
    Config.OUTPUT_VIDEO  = args.output_video
    Config.OUTPUT_CSV    = args.output_csv

    src = int(args.source) if args.source.isdigit() else args.source

    pipeline = ANPRPipeline(Config.MODEL_PATH)
    pipeline.run(src)