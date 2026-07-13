# HLS YOLO Frame Detector

Processes HLS (`.m3u8`) DVR streams: catches up from a historical offset, runs YOLO person detection on an ROI, saves matching full frames, then continues on the live edge. Multiple cameras run in parallel.

## Requirements

- Python 3.10+
- GPU recommended (CUDA) for YOLO; CPU works but is slower
- Network access to your HLS endpoints
- Model weights (e.g. `yolo26m.onnx` or `yolov8l.pt`) in the project directory or a path you set in config

## Setup

```bash
cd detect-frames
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your YOLO weights in the project root (or update `YOLO_MODEL` to the full path).

## Configure

All settings are edited at the top of `hls_yolo_detector.py`. There is no CLI.

### 1. Streams (cameras)

Add one dict per camera in `STREAMS`:

```python
STREAMS = [
    {
        "camera_id": "Faisal-Spinning",   # output folder name
        "url": "https://example.com/streams/cam1/live.m3u8",
        "roi_coords": [(828, 422), (444, 321), (226, 524), (596, 719)],
    },
    {
        "camera_id": "Akram-Textile",
        "url": "https://example.com/streams/cam2/live.m3u8",
        "roi_coords": [(100, 200), (600, 200), (600, 800), (100, 800)],
    },
]
```

| Field | Required | Description |
|--------|----------|-------------|
| `camera_id` | yes | Unique name; frames go to `output/<camera_id>/` |
| `url` | yes | Media playlist URL (`.m3u8`). Must return HTTP 200 |
| `roi_coords` | no | Per-camera ROI; falls back to global `ROI_COORDS` |

Verify each URL in a browser or with:

```bash
curl -I "https://your-host/streams/cam/live.m3u8"
```

A `404` means that camera will not process (check the log for `Failed to load playlist`).

### 2. ROI (region of interest)

ROI is a polygon of `(x, y)` pixel points. Origin `(0, 0)` is the **top-left** of the frame. A person is saved when their bounding box intersects this polygon.

**Global default** (used if a stream omits `roi_coords`):

```python
ROI_COORDS = [
    (200, 150),   # point 1
    (800, 150),   # point 2
    (800, 700),   # point 3
    (200, 700),   # point 4
]
```

**How to pick points**

1. Save one frame from the camera (or open a saved output JPEG).
2. Note corner coordinates of the area you care about (image viewer / annotation tool).
3. Paste at least 3 points, clockwise or counter-clockwise.
4. Prefer per-camera `roi_coords` when cameras have different views.

### 3. Model and detection

```python
YOLO_MODEL = "yolo26m.onnx"   # or yolov8l.pt, path/to/weights.pt
PERSON_CLASS_ID = 0           # COCO person
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
YOLO_DEVICE = 0               # GPU id: 0, 1, ... or "cpu"
```

### 4. Frame sampling

Stream is typically 20 FPS; inference can run at 1 FPS:

```python
SOURCE_FPS = 20
INFERENCE_FPS = 1             # → every 20th frame
```

### 5. Catch-up window

How far behind live to start (seconds):

```python
CATCHUP_OFFSET_SECONDS = 1 * 3600   # 1 hour ago
# CATCHUP_OFFSET_SECONDS = 4 * 3600 # 4 hours ago (2nd hour of a 6h DVR)
```

### 6. Download / disk

```python
DOWNLOAD_CONCURRENCY = 24    # parallel .ts downloads per camera
DOWNLOAD_WAVE_SIZE = 150     # segments per download wave before YOLO
# DOWNLOAD_WAVE_SIZE = 0     # entire catch-up list in one shot (~2MB × N disk)
```

Waves avoid filling the disk; the playlist is still held once (no re-fetch between segments inside catch-up).

### 7. Output paths

```python
OUTPUT_ROOT = Path("output")       # saved frames
TEMP_ROOT = Path("tmp_segments")   # temporary .ts files (deleted after use)
```

## Run

```bash
source .venv/bin/activate
python hls_yolo_detector.py
```

Stop with `Ctrl+C`.

### What you should see

1. Model load  
2. `Running N stream(s) in parallel: ...`  
3. Per camera: held m3u8 snapshot → `SINGLE-SHOT download` progress → YOLO progress / saved frames  
4. After catch-up: `live poll mode`

Saved frames:

```text
output/<camera_id>/<camera_id>_seqXXXXXXXXXX_fXXXXX_YYYYMMDD_HHMMSS.jpg
```

## Behavior summary

1. Load each camera’s `.m3u8` once and hold the segment list.  
2. Start from `CATCHUP_OFFSET_SECONDS` behind live.  
3. Download segments in waves (single-shot `asyncio.gather`), then run YOLO.  
4. Save full frames when a person intersects the ROI.  
5. Delete each `.ts` after processing.  
6. When historical segments are done, poll the live playlist for new segments.  
7. All cameras in `STREAMS` run concurrently (shared YOLO lock on GPU).

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Only one camera works | Other URL returns 404 or errors — check log for `Failed to load playlist` |
| `Download failed` / timeouts | Network or CDN limits; try lower `DOWNLOAD_CONCURRENCY` |
| `Segment gone (404)` | DVR segment expired; skipped automatically |
| No frames saved | ROI wrong, no persons, or confidence too high |
| GPU / ONNX errors | Set `YOLO_DEVICE = "cpu"` or fix CUDA / ONNX Runtime |
| Disk filling up | Lower `DOWNLOAD_WAVE_SIZE` (e.g. 50–150) |

## Optional: export ONNX

If you use `model.py`:

```bash
python model.py
```

Then point `YOLO_MODEL` at the exported `.onnx` file.
