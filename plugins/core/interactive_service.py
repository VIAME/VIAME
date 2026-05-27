#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Unified Interactive Service

Hosts the interactive segmentation backend (point/SAM segmentation, text query,
stereo point-segmentation) and the interactive stereo backend (enable, point/line
transfer, measurement, dense disparity) in a SINGLE process so they share one set
of loaded plugins and models. This lets stereo point-segmentation reuse the very
same stereo backend that interactive-stereo mode loaded, instead of loading a
second copy of the stereo model.

Each feature's models are loaded lazily on first use, so nothing heavy is loaded
unnecessarily:
  - segmentation models load on the first predict / text_query / stereo_segment
  - stereo models load on the first enable (or first stereo_segment)

Separate, unmodified VIAME config files drive each feature
(``--segmentation-config`` and ``--stereo-config``); the segmentation config
auto-discovers its text-query sibling exactly as the standalone service does.

Protocol is identical to the individual services: newline-delimited JSON requests
on stdin, JSON responses on stdout, with each response echoing the request ``id``.

Usage:
    python -m viame.core.interactive_service \\
        --segmentation-config /path/to/interactive_segmenter_default.conf \\
        --stereo-config /path/to/interactive_stereo_default.conf
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from viame.core.interactive_segmentation import (
    InteractiveSegmentationService,
    load_algorithms_from_config,
    find_viame_config as find_segmentation_config,
    suppress_stdout,
)
from viame.core.interactive_stereo import (
    InteractiveStereoService,
    load_algorithm_from_config,
    find_viame_config as find_stereo_config,
)


# Commands routed to the interactive-stereo backend. Everything else
# (predict, set_image, clear_image, text_query, refine, stereo_segment) is
# routed to the segmentation backend.
STEREO_COMMANDS = {
    "enable", "disable", "set_calibration", "set_frame", "cancel",
    "get_status", "transfer_line", "transfer_points", "measure_line",
    "aggregate_lengths",
}

# Stereo status/lifecycle commands that must NOT construct (load) the stereo
# backend on their own -- e.g. polling status on single-camera data, or
# disabling something that was never enabled, must stay cheap no-ops.
STEREO_IDLE_COMMANDS = {"get_status", "cancel", "disable"}

# Segmentation cache-management commands that must NOT construct (load) the
# segmentation backend / model before the user's first prediction.
SEG_IDLE_COMMANDS = {"set_image", "clear_image"}


class InteractiveService:
    """Single stdin/stdout loop that lazily hosts both backends."""

    def __init__(
        self,
        segmentation_configs: Optional[List[str]],
        stereo_config: Optional[str],
        plugin_paths: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self._segmentation_configs = segmentation_configs
        self._stereo_config = stereo_config
        self._plugin_paths = plugin_paths or []
        self._device = device

        # Lazily constructed sub-services (their models load lazily in turn).
        self._seg_service: Optional[InteractiveSegmentationService] = None
        self._stereo_service: Optional[InteractiveStereoService] = None

        self._build_lock = threading.Lock()  # guards lazy construction
        self._send_lock = threading.Lock()   # serializes stdout writes

    # ------------------------------------------------------------------ IO
    def _log(self, message: str) -> None:
        """Log to stderr (stdout is reserved for JSON responses)."""
        print(f"[InteractiveService] {message}", file=sys.stderr, flush=True)

    def _send(self, response: Dict[str, Any]) -> None:
        """Write one JSON response to stdout. Thread-safe: the stereo backend's
        background disparity worker and deferred-transfer threads also call this
        (via its injected send callback), so writes must be serialized."""
        with self._send_lock:
            print(json.dumps(response), flush=True)

    def _send_error(self, request_id: Optional[str], error: str) -> None:
        self._send({"id": request_id, "success": False, "error": error})

    # ------------------------------------------------- lazy construction
    def _ensure_segmentation(self) -> InteractiveSegmentationService:
        if self._seg_service is not None:
            return self._seg_service
        with self._build_lock:
            if self._seg_service is None:
                configs = self._segmentation_configs
                if not configs:
                    auto = find_segmentation_config()
                    configs = [auto] if auto else None
                if not configs:
                    raise ValueError(
                        "No segmentation config available "
                        "(interactive_segmenter_default.conf); is VIAME_INSTALL set?")
                self._log("Loading segmentation backend...")
                with suppress_stdout():
                    seg_algo, tq_algo, image_io_algo, svc_cfg = \
                        load_algorithms_from_config(
                            configs, self._plugin_paths, self._device)
                if seg_algo is None:
                    raise ValueError("No segment_via_points algorithm configured")
                self._seg_service = InteractiveSegmentationService(
                    segment_via_points_algo=seg_algo,
                    perform_text_query_algo=tq_algo,
                    image_io_algo=image_io_algo,
                    plugin_paths=self._plugin_paths,
                    device=self._device,
                    **svc_cfg,
                )
                self._log("Segmentation backend ready")
        return self._seg_service

    def _ensure_stereo(self) -> InteractiveStereoService:
        if self._stereo_service is not None:
            return self._stereo_service
        with self._build_lock:
            if self._stereo_service is None:
                config = self._stereo_config or find_stereo_config()
                if not config:
                    raise ValueError(
                        "No stereo config available "
                        "(interactive_stereo_default.conf); is VIAME_INSTALL set?")
                self._log("Loading stereo backend...")
                with suppress_stdout():
                    stereo_algo, matcher, svc_cfg = load_algorithm_from_config(
                        config, self._plugin_paths)
                if stereo_algo is None and matcher is None:
                    raise ValueError(
                        "No stereo algorithm or matching method configured")
                # Route the stereo backend's async (disparity) and deferred
                # output through our single locked writer.
                self._stereo_service = InteractiveStereoService(
                    compute_stereo_depth_map_algo=stereo_algo,
                    epipolar_matcher=matcher,
                    send_response=self._send,
                    **svc_cfg,
                )
                self._log(
                    "Stereo backend ready "
                    f"({'epipolar' if matcher is not None else 'dense'} mode)")
        return self._stereo_service

    # ----------------------------------------------------------- routing
    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route a request to the appropriate backend, building it on first use.

        Returns the response dict for synchronous commands, or None for stereo
        commands that defer their response to a background thread (which sends
        it later via the shared writer)."""
        command = request.get("command")

        if command in STEREO_COMMANDS:
            # Don't spin up the stereo backend just to answer a status/lifecycle
            # query (keeps single-camera sessions from ever loading stereo).
            if self._stereo_service is None and command in STEREO_IDLE_COMMANDS:
                return self._stereo_idle_response(command)
            return self._ensure_stereo().handle_request(request)

        # Build + warm up the segmentation models. Sent when the user enters
        # point-segmentation mode, so SAM loads on mode entry (not on the first
        # click). This is the ONLY segmentation command that loads the model
        # ahead of an actual prediction.
        if command == "init_segmentation":
            with suppress_stdout():
                self._ensure_segmentation().warmup()
            return {"success": True}

        # Segmentation side. Image cache management must not construct the
        # segmentation backend / load the model before the first prediction.
        if self._seg_service is None and command in SEG_IDLE_COMMANDS:
            return {"success": True}

        seg = self._ensure_segmentation()
        if command == "stereo_segment":
            # Reuse the one stereo backend (no second stereo model load).
            seg.set_stereo_warper(self._ensure_stereo())
        return seg.handle_request(request)

    @staticmethod
    def _stereo_idle_response(command: str) -> Dict[str, Any]:
        if command == "get_status":
            return {
                "success": True,
                "enabled": False,
                "disparity_ready": False,
                "has_calibration": False,
            }
        return {"success": True}

    # -------------------------------------------------------------- loop
    def run(self) -> None:
        # Nothing heavy at startup: plugin modules and per-feature models are
        # loaded lazily by the first relevant request (each backend's
        # load_*_from_config loads the plugins + plugin_paths it needs). This
        # keeps the process light until a feature is actually used, and lets the
        # stereo and segmentation methods initialize independently.
        self._log("Service started, waiting for requests...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")

                if request.get("command") == "shutdown":
                    self._log("Shutdown requested")
                    self._shutdown_backends()
                    self._send({
                        "id": request_id,
                        "success": True,
                        "message": "Shutting down",
                    })
                    break

                response = self.handle_request(request)
                if response is not None:
                    response["id"] = request_id
                    self._send(response)

            except json.JSONDecodeError as e:
                self._send_error(request_id, f"Invalid JSON: {e}")
            except Exception as e:
                self._log(f"Error processing request: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._send_error(request_id, str(e))

        self._log("Service shutting down")

    def _shutdown_backends(self) -> None:
        """Stop the stereo backend's background worker thread cleanly."""
        if self._stereo_service is not None:
            try:
                self._stereo_service._cancel_computation()
                if self._stereo_service._compute_queue is not None:
                    self._stereo_service._compute_queue.put(None)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Unified Interactive Service (segmentation + stereo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segmentation-config",
        action="append",
        default=None,
        help="Segmentation KWIVER config file(s). May be specified more than "
             "once (e.g. segmenter + text query). If omitted, the default "
             "interactive segmenter config is auto-detected.",
    )
    parser.add_argument(
        "--stereo-config",
        default=None,
        help="Interactive stereo KWIVER config file. If omitted, "
             "interactive_stereo_default.conf is auto-detected.",
    )
    parser.add_argument(
        "--plugin-path",
        action="append",
        default=[],
        help="Additional plugin paths to load (can be specified multiple times)",
    )
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME install directory (sets VIAME_INSTALL env var)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda, cpu, auto)",
    )
    args = parser.parse_args()

    if args.viame_path:
        os.environ["VIAME_INSTALL"] = args.viame_path

    for c in (args.segmentation_config or []):
        if not Path(c).exists():
            print(f"Error: segmentation config not found: {c}", file=sys.stderr)
            sys.exit(1)
    if args.stereo_config and not Path(args.stereo_config).exists():
        print(f"Error: stereo config not found: {args.stereo_config}",
              file=sys.stderr)
        sys.exit(1)

    service = InteractiveService(
        segmentation_configs=args.segmentation_config,
        stereo_config=args.stereo_config,
        plugin_paths=args.plugin_path,
        device=args.device,
    )

    try:
        service.run()
    except KeyboardInterrupt:
        print("[InteractiveService] Interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[InteractiveService] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
