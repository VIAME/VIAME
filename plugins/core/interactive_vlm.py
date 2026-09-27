#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Vision-language model (VLM) queries against a locally served model (Ollama).

Used by the interactive service:
  - vlm_detect: text-prompted grounding, returning boxes in the text-query
    detection format
  - vlm_ask: free-form question about one or more frames or track crops
and by the ``ollama_vlm`` track refiner for whole-sequence text queries.
"""

import base64
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_OLLAMA_PORT = 11434
MAX_IMAGE_SIDE = 1280
CROP_PADDING = 0.15
REQUEST_TIMEOUT = 290

VIDEO_EXTENSIONS = {
    '.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpg', '.mpeg', '.m4v'}

DETECT_PROMPT = (
    'Locate every instance of "{text}" in the image. Output a JSON list where '
    'each entry has "bbox_2d" [x1, y1, x2, y2] and a short "label". '
    'Output [] if there are none.'
)


def ollama_url(host: Optional[str] = None) -> str:
    """Base URL of the Ollama server, read from OLLAMA_HOST as the Ollama CLI does."""
    host = (host if host is not None else os.environ.get("OLLAMA_HOST", "")).strip()
    if not host:
        return f"http://localhost:{DEFAULT_OLLAMA_PORT}"
    has_scheme = re.match(r"^https?://", host, flags=re.IGNORECASE) is not None
    parsed = urllib.parse.urlsplit(host if has_scheme else f"http://{host}")
    hostname = parsed.hostname or "localhost"
    if hostname == "0.0.0.0":
        hostname = "localhost"
    port = parsed.port or (None if has_scheme else DEFAULT_OLLAMA_PORT)
    netloc = f"{hostname}:{port}" if port else hostname
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", "")).rstrip("/")


def ollama_chat(
    model: str,
    messages: List[Dict[str, Any]],
    think: Optional[bool] = None,
    temperature: float = 0,
) -> str:
    url = ollama_url()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if think is not None:
        payload["think"] = bool(think)
    http_request = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=REQUEST_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        if e.code == 404:
            detail = f"{detail}. Download it with: ollama pull {model}"
        raise RuntimeError(f"Ollama error ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {url} ({e.reason}). "
            "Is the Ollama service running?") from e
    return body.get("message", {}).get("content", "")


def encode_image(image) -> str:
    """PIL image -> base64 JPEG, downscaled so the long side fits the model input."""
    scale = MAX_IMAGE_SIDE / max(image.width, image.height)
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_image(image, box: List[float]):
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * CROP_PADDING
    pad_y = (y2 - y1) * CROP_PADDING
    return image.crop((
        max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
        min(image.width, int(x2 + pad_x)), min(image.height, int(y2 + pad_y)),
    ))


def detect_objects(
    image, text: str, model: str, think: Optional[bool] = None,
) -> List[Tuple[List[float], Optional[str]]]:
    """Ground ``text`` in a PIL image; returns (pixel box, label) pairs."""
    content = ollama_chat(model, [{
        "role": "user",
        "content": DETECT_PROMPT.format(text=text),
        "images": [encode_image(image)],
    }], think=think)
    return parse_grounding(content, image.width, image.height)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _extract_json(text: str) -> Any:
    text = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except ValueError:
        return None


def parse_grounding(
    text: str, width: int, height: int,
) -> List[Tuple[List[float], Optional[str]]]:
    """Parse VLM grounding output into pixel boxes.

    Qwen-VL grounding coordinates are normalized to 0-1000 regardless of the
    input resolution, so they are rescaled to the source frame size.
    """
    data = _extract_json(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    if not isinstance(data, list):
        return []

    results = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        coords = entry.get("bbox_2d") or entry.get("bbox") or entry.get("box")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(c) for c in coords)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        box = [
            min(max(x1, 0.0), 1000.0) * width / 1000.0,
            min(max(y1, 0.0), 1000.0) * height / 1000.0,
            min(max(x2, 0.0), 1000.0) * width / 1000.0,
            min(max(y2, 0.0), 1000.0) * height / 1000.0,
        ]
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        label = entry.get("label")
        results.append((box, str(label).strip() if label else None))
    return results


class VlmService:
    """Stateless apart from a cached video reader; safe to build eagerly."""

    COMMANDS = {"vlm_detect", "vlm_ask"}

    def __init__(self):
        self._video_reader = None
        self._video_reader_path = None
        self._video_fps = None

    def _log(self, message: str) -> None:
        print(f"[VlmService] {message}", file=sys.stderr, flush=True)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = request.get("command")
        if command == "vlm_detect":
            return self.handle_detect(request)
        if command == "vlm_ask":
            return self.handle_ask(request)
        raise ValueError(f"Unknown VLM command: {command}")

    def _load_frame(self, image_path: str, frame_time: Optional[float]):
        from PIL import Image

        ext = os.path.splitext(image_path)[1].lower()
        if ext in VIDEO_EXTENSIONS and frame_time is not None:
            array = self._load_video_frame(image_path, frame_time)
            return Image.fromarray(array).convert("RGB")
        return Image.open(image_path).convert("RGB")

    def _load_video_frame(self, video_path: str, frame_time: float):
        from kwiver.vital.algo import VideoInput

        if self._video_reader is None or self._video_reader_path != video_path:
            reader = VideoInput.create("vidl_ffmpeg")
            cfg = reader.get_configuration()
            cfg.set_value("time_source", "start_at_0")
            reader.set_configuration(cfg)
            reader.open(video_path)
            reader.next_frame()
            self._video_reader = reader
            self._video_reader_path = video_path
            self._video_fps = reader.frame_rate()

        # vidl_ffmpeg frame numbers are 1-based
        target_frame = max(1, round(frame_time * self._video_fps) + 1)
        self._video_reader.seek_frame(target_frame)
        image = self._video_reader.frame_image()
        if image is None:
            raise RuntimeError(
                f"Could not read frame at t={frame_time:.3f}s from {video_path}")
        return image.image().asarray()

    def handle_detect(self, request: Dict[str, Any]) -> Dict[str, Any]:
        image_path = request.get("image_path")
        text = (request.get("text") or "").strip()
        if not image_path:
            raise ValueError("image_path is required")
        if not text:
            raise ValueError("text query is required")
        if not request.get("model"):
            raise ValueError("model is required")

        image = self._load_frame(image_path, request.get("frame_time"))
        self._log(f"detect '{text}' with {request['model']} on {os.path.basename(image_path)}")
        boxes = detect_objects(image, text, request["model"], request.get("think"))
        max_detections = request.get("max_detections") or len(boxes)
        detections = [{
            "box": box,
            "bounds": box,
            "score": 1.0,
            "label": label or text,
        } for box, label in boxes[:max_detections]]
        return {"success": True, "detections": detections, "query": text}

    def handle_ask(self, request: Dict[str, Any]) -> Dict[str, Any]:
        question = (request.get("question") or "").strip()
        specs = request.get("images") or []
        if not question:
            raise ValueError("question is required")
        if not specs:
            raise ValueError("at least one image is required")
        if not request.get("model"):
            raise ValueError("model is required")

        images = []
        for spec in specs:
            image = self._load_frame(spec["image_path"], spec.get("frame_time"))
            if spec.get("box"):
                image = crop_image(image, spec["box"])
            images.append(encode_image(image))

        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in request.get("history") or []
        ]
        messages.append({"role": "user", "content": question, "images": images})
        self._log(f"ask with {request['model']} over {len(images)} image(s)")
        answer = ollama_chat(request["model"], messages, think=request.get("think"))
        return {"success": True, "answer": strip_thinking(answer).strip()}
