#!/usr/bin/env python3
"""
HLS catch-up + live YOLO person detector.

Catch-up strategy (single-shot, industry-standard HLS bulk download):
  1. Fetch index.m3u8 ONCE and hold the full segment URL list in memory.
  2. Slice from 4 hours behind live to the snapshot edge.
  3. Download that URL list with asyncio.gather + a connection pool
     (same pattern as aiom3u8downloader / m3u8-dl). No playlist re-fetch
     between segments. YOLO does not run during a download wave.
  4. After a wave is fully on disk, feed segments to YOLO in order.
  5. When the held list is exhausted, poll the live m3u8 for new segments.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import cv2
import m3u8
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration — edit these for your deployment
# ---------------------------------------------------------------------------

ROI_COORDS: List[Tuple[int, int]] = [
    (200, 150),
    (800, 150),
    (800, 700),
    (200, 700),
]

STREAMS: List[dict] = [
    {
        "camera_id": "Faisal-Spinning",
        "url": "https://faisal-stream.betacodespk.com/streams/FaisalSpiningcamera1/live.m3u8",
        "roi_coords": [(828, 422), (444, 321), (226, 524), (596, 719)],
    },
    {
        "camera_id": "Akram-Textile",
        "url": "https://akram-stream.betacodespk.com/streams/AkramCamera1/live.m3u8",
        "roi_coords": [(828, 422), (444, 321), (226, 524), (596, 719)],
    },
]

YOLO_MODEL = "yolo26m.onnx"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
YOLO_DEVICE = 1

SOURCE_FPS = 20
INFERENCE_FPS = 1
FRAME_STRIDE = max(1, int(round(SOURCE_FPS / INFERENCE_FPS)))

CATCHUP_OFFSET_SECONDS = 1 * 3600

# Single-shot bulk download settings (aiom3u8downloader-style).
# Concurrency > ~24 does NOT help on this CDN — pipe caps ~1.2 MB/s.
DOWNLOAD_CONCURRENCY = 24
DOWNLOAD_RETRIES = 3  # transient errors only; 404 fails immediately
DOWNLOAD_RETRY_BASE_DELAY = 0.4
DOWNLOAD_TIMEOUT_TOTAL = 90
DOWNLOAD_TIMEOUT_CONNECT = 10
DOWNLOAD_TIMEOUT_READ = 60

# How many segments to download in one gather() before running YOLO on them.
# 0 = download the ENTIRE catch-up list in one shot (~2MB * N disk).
# 150 ≈ 5 minutes of video per wave — safer on disk, still no m3u8 re-fetch.
DOWNLOAD_WAVE_SIZE = 150

QUEUE_MAXSIZE = 8
LIVE_POLL_INTERVAL = 1.0

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

OUTPUT_ROOT = Path("output")
TEMP_ROOT = Path("tmp_segments")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hls_yolo")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentInfo:
    uri: str
    media_sequence: int
    duration: float
    absolute_uri: str

    @property
    def key(self) -> str:
        return f"{self.media_sequence}:{self.uri}"


@dataclass
class QueuedSegment:
    info: SegmentInfo
    path: Path
    camera_id: str


@dataclass
class StreamConfig:
    camera_id: str
    url: str
    roi_coords: List[Tuple[int, int]] = field(default_factory=lambda: list(ROI_COORDS))
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = OUTPUT_ROOT / self.camera_id


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def roi_to_numpy(roi_coords: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(roi_coords) < 3:
        raise ValueError("ROI must contain at least 3 points")
    return np.array(roi_coords, dtype=np.int32)


def bbox_intersects_roi(
    bbox: Tuple[float, float, float, float],
    roi_poly: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return False
    rx, ry, rw, rh = cv2.boundingRect(roi_poly)
    if x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh:
        return False
    pad = 2
    min_x = max(0, min(x1, rx) - pad)
    min_y = max(0, min(y1, ry) - pad)
    max_x = max(x2, rx + rw) + pad
    max_y = max(y2, ry + rh) + pad
    w, h = max_x - min_x + 1, max_y - min_y + 1
    if w <= 0 or h <= 0:
        return False
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    mask_box = np.zeros((h, w), dtype=np.uint8)
    shifted = roi_poly - np.array([min_x, min_y])
    cv2.fillPoly(mask_roi, [shifted], 255)
    cv2.rectangle(
        mask_box,
        (x1 - min_x, y1 - min_y),
        (x2 - min_x, y2 - min_y),
        255,
        thickness=-1,
    )
    return bool(cv2.countNonZero(cv2.bitwise_and(mask_roi, mask_box)))


# ---------------------------------------------------------------------------
# M3U8 parsing & time navigation
# ---------------------------------------------------------------------------

class PlaylistParser:
    """Parse HLS playlists and compute the catch-up start offset."""

    def __init__(self, playlist_url: str) -> None:
        self.playlist_url = playlist_url
        self.base_uri = self._base_uri(playlist_url)

    @staticmethod
    def _base_uri(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rsplit("/", 1)[0] + "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def load(self) -> m3u8.M3U8:
        playlist = m3u8.load(self.playlist_url)
        if playlist.is_variant and playlist.playlists:
            best = max(
                playlist.playlists,
                key=lambda p: (p.stream_info.bandwidth or 0),
            )
            media_url = urljoin(self.playlist_url, best.uri)
            logger.info("Resolved variant playlist → %s", media_url)
            self.playlist_url = media_url
            self.base_uri = self._base_uri(media_url)
            playlist = m3u8.load(media_url)
        return playlist

    def segments_from_playlist(self, playlist: m3u8.M3U8) -> List[SegmentInfo]:
        media_seq = playlist.media_sequence or 0
        out: List[SegmentInfo] = []
        for i, seg in enumerate(playlist.segments):
            abs_uri = seg.absolute_uri or urljoin(self.base_uri, seg.uri)
            out.append(
                SegmentInfo(
                    uri=seg.uri,
                    media_sequence=media_seq + i,
                    duration=float(seg.duration or 0.0),
                    absolute_uri=abs_uri,
                )
            )
        return out

    def find_catchup_index(
        self,
        segments: Sequence[SegmentInfo],
        offset_seconds: float = CATCHUP_OFFSET_SECONDS,
    ) -> int:
        if not segments:
            return 0
        accumulated = 0.0
        start_idx = 0
        for i in range(len(segments) - 1, -1, -1):
            accumulated += segments[i].duration
            if accumulated >= offset_seconds:
                start_idx = i
                break
        else:
            start_idx = 0
            logger.warning(
                "Playlist spans only %.1fs (< %.1fs requested); starting at oldest",
                accumulated,
                offset_seconds,
            )
        logger.info(
            "Catch-up start: segment[%d] seq=%d (lookback≈%.1fs of %.1fs available)",
            start_idx,
            segments[start_idx].media_sequence,
            min(accumulated, offset_seconds) if accumulated else 0.0,
            sum(s.duration for s in segments),
        )
        return start_idx


# ---------------------------------------------------------------------------
# Single-shot bulk segment downloader
# ---------------------------------------------------------------------------

class BulkSegmentDownloader:
    """
    Download an entire held segment list in one shot.

    Pattern (same as aiom3u8downloader / m3u8-dl):
      - Take a list of absolute .ts URLs (already parsed from one m3u8).
      - Fire asyncio.gather over all of them.
      - Cap concurrency with TCPConnector + Semaphore.
      - Do NOT re-fetch the playlist between segments.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        dest_dir: Path,
        concurrency: int = DOWNLOAD_CONCURRENCY,
        retries: int = DOWNLOAD_RETRIES,
    ) -> None:
        self.session = session
        self.dest_dir = dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.retries = retries
        self.timeout = aiohttp.ClientTimeout(
            total=DOWNLOAD_TIMEOUT_TOTAL,
            sock_connect=DOWNLOAD_TIMEOUT_CONNECT,
            sock_read=DOWNLOAD_TIMEOUT_READ,
        )

    def _path_for(self, segment: SegmentInfo) -> Path:
        return self.dest_dir / f"seg_{segment.media_sequence:010d}.ts"

    async def _download_one(self, segment: SegmentInfo) -> Optional[Path]:
        dest = self._path_for(segment)
        if dest.exists() and dest.stat().st_size > 0:
            return dest

        last_err: Optional[BaseException] = None
        async with self.semaphore:
            for attempt in range(1, self.retries + 1):
                try:
                    async with self.session.get(
                        segment.absolute_uri,
                        timeout=self.timeout,
                    ) as resp:
                        # Expired DVR segments — fail fast, do not burn retries
                        if resp.status == 404:
                            logger.warning(
                                "Segment gone (404) seq=%d %s — skipping",
                                segment.media_sequence,
                                segment.uri,
                            )
                            return None
                        resp.raise_for_status()
                        tmp = dest.with_suffix(".partial")
                        size = 0
                        with open(tmp, "wb") as fh:
                            async for chunk in resp.content.iter_chunked(64 * 1024):
                                fh.write(chunk)
                                size += len(chunk)
                        if size <= 0:
                            raise IOError("empty segment body")
                        tmp.replace(dest)
                        return dest
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    last_err = exc
                    err = str(exc).strip() or type(exc).__name__
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Download seq=%d attempt=%d/%d: %s — retry in %.1fs",
                        segment.media_sequence,
                        attempt,
                        self.retries,
                        err,
                        delay,
                    )
                    await asyncio.sleep(delay)

        logger.error(
            "Giving up seq=%d after %d attempts: %s",
            segment.media_sequence,
            self.retries,
            last_err,
        )
        return None

    async def download_all(
        self,
        segments: Sequence[SegmentInfo],
        label: str = "bulk",
    ) -> Dict[int, Path]:
        """
        Single-shot download of every segment in `segments`.

        Returns {media_sequence: local_path} for successes only.
        """
        if not segments:
            return {}

        total = len(segments)
        done = 0
        bytes_total = 0
        t0 = time.monotonic()
        results: Dict[int, Path] = {}
        lock = asyncio.Lock()

        logger.info(
            "[%s] SINGLE-SHOT download starting: %d segments "
            "(concurrency=%d, no m3u8 re-fetch)",
            label,
            total,
            DOWNLOAD_CONCURRENCY,
        )

        async def _one(seg: SegmentInfo) -> None:
            nonlocal done, bytes_total
            path = await self._download_one(seg)
            async with lock:
                if path is not None:
                    results[seg.media_sequence] = path
                    bytes_total += path.stat().st_size
                done += 1
                if done == 1 or done == total or done % 25 == 0:
                    elapsed = time.monotonic() - t0
                    seg_s = done / elapsed if elapsed > 0 else 0.0
                    mb_s = (bytes_total / 1e6) / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "[%s] download %d/%d (%.1f seg/s, %.2f MB/s, ok=%d)",
                        label,
                        done,
                        total,
                        seg_s,
                        mb_s,
                        len(results),
                    )

        # Fire the entire wave at once — connector/semaphore cap concurrency.
        await asyncio.gather(*[_one(seg) for seg in segments])

        elapsed = time.monotonic() - t0
        logger.info(
            "[%s] SINGLE-SHOT done: ok=%d/%d in %.1fs (%.1f seg/s, %.2f MB/s)",
            label,
            len(results),
            total,
            elapsed,
            len(results) / elapsed if elapsed > 0 else 0.0,
            (bytes_total / 1e6) / elapsed if elapsed > 0 else 0.0,
        )
        return results


def _chunked(items: Sequence[SegmentInfo], size: int) -> Iterable[Sequence[SegmentInfo]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Producer — hold m3u8, single-shot download waves, then feed YOLO
# ---------------------------------------------------------------------------

class SegmentProducer:
    """
    1) Hold one m3u8 snapshot.
    2) Single-shot download each wave from that held list.
    3) Only after a wave is on disk, enqueue it for YOLO.
    4) After the held list is done, poll live m3u8 for new segments.
    """

    def __init__(
        self,
        config: StreamConfig,
        queue: asyncio.Queue,
        offset_seconds: float = CATCHUP_OFFSET_SECONDS,
    ) -> None:
        self.config = config
        self.queue = queue
        self.offset_seconds = offset_seconds
        self.parser = PlaylistParser(config.url)
        self._seen: set[str] = set()
        self._stop = asyncio.Event()
        self.temp_dir = TEMP_ROOT / config.camera_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=DOWNLOAD_CONCURRENCY,
            limit_per_host=DOWNLOAD_CONCURRENCY,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=DOWNLOAD_TIMEOUT_CONNECT,
            sock_read=DOWNLOAD_TIMEOUT_READ,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=HTTP_HEADERS,
        ) as session:
            downloader = BulkSegmentDownloader(session, self.temp_dir)

            # ---- Hold ONE playlist snapshot ----
            try:
                playlist = await asyncio.to_thread(self.parser.load)
            except Exception as exc:
                logger.error(
                    "[%s] Failed to load playlist %s: %s",
                    self.config.camera_id,
                    self.config.url,
                    exc,
                )
                await self.queue.put(None)
                return

            snapshot = self.parser.segments_from_playlist(playlist)
            if not snapshot:
                logger.error("[%s] Empty playlist", self.config.camera_id)
                await self.queue.put(None)
                return

            start_idx = self.parser.find_catchup_index(snapshot, self.offset_seconds)
            historical = snapshot[start_idx:]
            wave_size = DOWNLOAD_WAVE_SIZE if DOWNLOAD_WAVE_SIZE > 0 else len(historical)
            n_waves = (len(historical) + wave_size - 1) // wave_size
            logger.info(
                "[%s] Held m3u8 snapshot: %d catch-up segments "
                "(seq %d→%d) in %d single-shot wave(s) of ≤%d",
                self.config.camera_id,
                len(historical),
                historical[0].media_sequence,
                historical[-1].media_sequence,
                n_waves,
                wave_size,
            )

            for wave_i, wave in enumerate(_chunked(historical, wave_size), start=1):
                if self._stop.is_set():
                    break
                label = f"{self.config.camera_id}/catch-up-wave{wave_i}/{n_waves}"
                await self._download_wave_then_enqueue(downloader, wave, label)

            # ---- Live: poll m3u8 only after held snapshot is exhausted ----
            logger.info("[%s] Catch-up done — live poll mode", self.config.camera_id)
            while not self._stop.is_set():
                try:
                    playlist = await asyncio.to_thread(self.parser.load)
                    live_segs = self.parser.segments_from_playlist(playlist)
                    new_segs = [s for s in live_segs if s.key not in self._seen]
                    if new_segs:
                        label = f"{self.config.camera_id}/live"
                        await self._download_wave_then_enqueue(
                            downloader, new_segs, label
                        )
                except Exception:
                    logger.exception("[%s] Live poll error", self.config.camera_id)
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=LIVE_POLL_INTERVAL
                    )
                except asyncio.TimeoutError:
                    pass

            await self.queue.put(None)

    async def _download_wave_then_enqueue(
        self,
        downloader: BulkSegmentDownloader,
        wave: Sequence[SegmentInfo],
        label: str,
    ) -> None:
        """Phase 1: single-shot download. Phase 2: enqueue for YOLO in order."""
        to_fetch = [s for s in wave if s.key not in self._seen]
        if not to_fetch:
            return

        # PHASE 1 — download only (no YOLO)
        paths = await downloader.download_all(to_fetch, label=label)
        if self._stop.is_set():
            return

        # PHASE 2 — hand off to consumer in playlist order, then wait for drain
        queued = 0
        for seg in to_fetch:
            self._seen.add(seg.key)
            path = paths.get(seg.media_sequence)
            if path is None:
                continue
            await self.queue.put(
                QueuedSegment(info=seg, path=path, camera_id=self.config.camera_id)
            )
            queued += 1

        if queued:
            logger.info(
                "[%s] Wave queued %d segments for YOLO — waiting for drain",
                label,
                queued,
            )
            await self.queue.join()
            logger.info("[%s] YOLO drained wave", label)


# ---------------------------------------------------------------------------
# YOLO inference + ROI (Consumer)
# ---------------------------------------------------------------------------

class YOLODetector:
    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        # Serialize GPU inference when multiple cameras run in parallel
        self._infer_lock = threading.Lock()
        logger.info("Loaded YOLO model: %s", model_path)

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        with self._infer_lock:
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device=self.device,
            )
        boxes: List[Tuple[float, float, float, float]] = []
        if not results:
            return boxes
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return boxes
        xyxy = r0.boxes.xyxy.cpu().numpy()
        for row in xyxy:
            boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        return boxes


class SegmentConsumer:
    def __init__(
        self,
        config: StreamConfig,
        queue: asyncio.Queue,
        detector: YOLODetector,
    ) -> None:
        self.config = config
        self.queue = queue
        self.detector = detector
        self.roi_poly = roi_to_numpy(config.roi_coords)
        self.output_dir = config.output_dir or (OUTPUT_ROOT / config.camera_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_saved = 0
        self.segments_processed = 0

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    logger.info(
                        "[%s] Consumer done — segments=%d frames_saved=%d",
                        self.config.camera_id,
                        self.segments_processed,
                        self.frames_saved,
                    )
                    return
                await asyncio.to_thread(self._process_segment, item)
            finally:
                self.queue.task_done()

    def _process_segment(self, item: QueuedSegment) -> None:
        path = item.path
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.error(
                    "[%s] Cannot open seq=%d (%s)",
                    item.camera_id,
                    item.info.media_sequence,
                    path,
                )
                return

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % FRAME_STRIDE != 0:
                    frame_idx += 1
                    continue
                persons = self.detector.detect_persons(frame)
                hit = any(bbox_intersects_roi(b, self.roi_poly) for b in persons)
                if hit:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_name = (
                        f"{item.camera_id}_seq{item.info.media_sequence:010d}"
                        f"_f{frame_idx:05d}_{ts}.jpg"
                    )
                    out_path = self.output_dir / out_name
                    cv2.imwrite(str(out_path), frame)
                    self.frames_saved += 1
                    logger.info(
                        "[%s] Saved frame → %s (persons=%d)",
                        item.camera_id,
                        out_path,
                        len(persons),
                    )
                frame_idx += 1

            cap.release()
            self.segments_processed += 1
            if self.segments_processed % 25 == 0:
                logger.info(
                    "[%s] Progress: %d segments, %d frames saved",
                    item.camera_id,
                    self.segments_processed,
                    self.frames_saved,
                )
        finally:
            self._safe_unlink(path)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def process_stream(
    config: StreamConfig,
    detector: YOLODetector,
    offset_seconds: float = CATCHUP_OFFSET_SECONDS,
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    producer = SegmentProducer(config, queue, offset_seconds=offset_seconds)
    consumer = SegmentConsumer(config, queue, detector)

    logger.info(
        "Starting stream camera_id=%s url=%s offset=%ss",
        config.camera_id,
        config.url,
        offset_seconds,
    )

    prod_task = asyncio.create_task(producer.run(), name=f"prod-{config.camera_id}")
    cons_task = asyncio.create_task(consumer.run(), name=f"cons-{config.camera_id}")

    try:
        await asyncio.gather(prod_task, cons_task)
    except asyncio.CancelledError:
        producer.stop()
        raise
    finally:
        producer.stop()
        cam_tmp = TEMP_ROOT / config.camera_id
        if cam_tmp.exists():
            shutil.rmtree(cam_tmp, ignore_errors=True)


async def process_all_streams(
    streams: Iterable[dict],
    model_path: str = YOLO_MODEL,
    offset_seconds: float = CATCHUP_OFFSET_SECONDS,
    device: Optional[str] = None,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    detector = YOLODetector(model_path=model_path, device=device)

    configs = [
        StreamConfig(
            camera_id=entry["camera_id"],
            url=entry["url"],
            roi_coords=list(entry.get("roi_coords", ROI_COORDS)),
        )
        for entry in streams
    ]
    logger.info(
        "Running %d stream(s) in parallel: %s",
        len(configs),
        ", ".join(c.camera_id for c in configs),
    )

    # Live mode never ends, so cameras must run concurrently — not one-after-another.
    tasks = [
        asyncio.create_task(
            process_stream(cfg, detector, offset_seconds=offset_seconds),
            name=f"stream-{cfg.camera_id}",
        )
        for cfg in configs
    ]

    # Surface per-stream failures immediately (don't wait for other cams to finish).
    async def _watch(cfg: StreamConfig, task: asyncio.Task) -> None:
        try:
            await task
        except Exception:
            logger.exception("[%s] Stream crashed", cfg.camera_id)

    await asyncio.gather(
        *[_watch(cfg, task) for cfg, task in zip(configs, tasks)]
    )

def main() -> None:
    if not STREAMS:
        logger.error("STREAMS is empty — add at least one camera URL in the script.")
        raise SystemExit(2)

    try:
        asyncio.run(
            process_all_streams(
                STREAMS,
                model_path=YOLO_MODEL,
                offset_seconds=CATCHUP_OFFSET_SECONDS,
                device=YOLO_DEVICE,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


if __name__ == "__main__":
    main()
