#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Video Query / IQR Service

Hosts VIAME's video search and iterative query refinement (IQR) engine as a
persistent service for GUI frontends (DIVE desktop). Wraps the
``query_retrieval_and_iqr.pipe`` KWIVER pipeline in an embedded pipeline and
manages the per-index embedded PostgreSQL instance, mirroring the protocol the
VIQUI (vivia) KIP query session speaks:

  input adapter ports : descriptor_request, database_query, iqr_feedback,
                        iqr_model  (exactly one populated per step; the rest
                        carry typed null sptrs)
  output adapter ports: track_descriptor_set, query_result, feedback_request,
                        iqr_model

An "index" is a directory containing the ``database/`` folder produced by
``process_video.py --build-index`` (embedded PostgreSQL data at
``database/SQL``, ITQ/LSH files at ``database/ITQ``). The pipeline's config
paths are CWD-relative, so this service chdirs into the index directory when
opening it; one index may be open at a time.

Protocol: newline-delimited JSON requests on stdin, JSON responses on stdout,
each response echoing the request ``id``. Commands:

  open_index      {index_dir, pipeline_file?}
  close_index     {}
  status          {}
  formulate_query {image_path, boxes?: [[x1,y1,x2,y2],...]}
      Computes exemplar descriptors from an image chip. When boxes are given
      the pipeline also auto-runs an initial similarity query in the same
      step, so the response may already carry ``results``.
  process_query   {threshold?, iqr_model_b64?}
      Runs (or re-runs) the similarity query from the last formulation's
      descriptors, optionally warm-started from a saved IQR model.
  refine          {positive_ids: [int], negative_ids: [int]}
      One IQR iteration: feedback + previous model in, re-ranked results and
      an updated model out.
  export_model    {output_path?}
      Returns the current model as base64; also writes it if a path is given.
  shutdown        {}

Result entries are serialized as
  {instance_id, stream_id, relevancy_score, start_frame, end_frame,
   tracks: [{id, states: [{frame, bbox: [x1,y1,x2,y2]}]}]}

Usage:
    python -m viame.core.query_service [--pipeline-file <query pipe>]
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


def _log(message: str) -> None:
    """Log to stderr (stdout is reserved for JSON responses)."""
    print(f"[QueryService] {message}", file=sys.stderr, flush=True)


def _is_windows() -> bool:
    return os.name == "nt"


def _exe(cmd: str) -> str:
    return cmd + ".exe" if _is_windows() else cmd


# --------------------------------------------------------------------------
# Embedded PostgreSQL lifecycle (per-index database/SQL data directory).
#
# Intentionally NOT reusing database_tool.stop(), which pkill -9's every
# postgres process on the machine; a desktop service must only touch the
# instance belonging to its own data directory.
# --------------------------------------------------------------------------
class PostgresInstance:
    SQL_DIR = os.path.join("database", "SQL")
    LOG_FILE = os.path.join("database", "SQL_Log_File")
    PORT = 5432

    @classmethod
    def _pg_ctl(cls, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [_exe("pg_ctl"), "-D", cls.SQL_DIR] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    @classmethod
    def is_running(cls) -> bool:
        return cls._pg_ctl(["status"]).returncode == 0

    @classmethod
    def start(cls) -> None:
        if cls.is_running():
            return
        # Recover from a previous unclean shutdown (stale postmaster.pid with
        # no live server behind it); pg_ctl start handles most cases itself,
        # but a pid file pointing at a recycled pid can block startup.
        pid_file = os.path.join(cls.SQL_DIR, "postmaster.pid")
        result = cls._pg_ctl(
            ["-w", "-t", "20", "-l", cls.LOG_FILE, "start"])
        if result.returncode != 0 and os.path.exists(pid_file):
            _log("postgres start failed; removing stale postmaster.pid "
                 "and retrying")
            os.remove(pid_file)
            result = cls._pg_ctl(
                ["-w", "-t", "20", "-l", cls.LOG_FILE, "start"])
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to start postgres for index: {result.stdout}")

    @classmethod
    def stop(cls) -> None:
        cls._pg_ctl(["-m", "fast", "stop"])
        cls._wait_for_port_available()

    @classmethod
    def _wait_for_port_available(cls, timeout: float = 10.0) -> bool:
        import socket
        deadline = time.time() + timeout
        while time.time() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", cls.PORT))
                sock.close()
                return True
            except OSError:
                sock.close()
                time.sleep(0.5)
        return False


# --------------------------------------------------------------------------
# Query session: one open index + embedded pipeline + evolving IQR model
# --------------------------------------------------------------------------
class QuerySession:
    """Wraps the embedded query pipeline for one open index directory."""

    INPUT_PORTS = ("descriptor_request", "database_query",
                   "iqr_feedback", "iqr_model")

    def __init__(self, index_dir: str, pipeline_file: str):
        # Import lazily so --help etc. work without a full VIAME environment.
        from kwiver.sprokit.adapters import adapter_data_set, embedded_pipeline
        from kwiver.vital import types as kvt

        self._ads = adapter_data_set
        self._kvt = kvt

        self.index_dir = os.path.abspath(index_dir)
        self.pipeline_file = pipeline_file

        # State carried across the IQR loop (mirrors vvKipQuerySession)
        self._descriptors: List[Any] = []      # last formulated exemplars
        self._query_id: str = ""
        self._query_model: Optional[Any] = None     # warm-start VectorUChar
        self._computed_model: Optional[Any] = None  # backend-owned model
        self._formulation_count = 0

        if not os.path.isdir(
                os.path.join(self.index_dir, "database", "ITQ")):
            raise ValueError(
                f"{index_dir} does not contain a built search index "
                "(missing database/ITQ)")

        # The pipeline resolves database_folder etc. against the CWD.
        os.chdir(self.index_dir)
        PostgresInstance.start()

        _log(f"Building query pipeline from {pipeline_file}")
        self._pipeline = embedded_pipeline.EmbeddedPipeline()
        self._pipeline.build_pipeline(pipeline_file)
        self._pipeline.start()
        _log(f"Index opened: {self.index_dir}")

    # -------------------------------------------------------------- helpers
    def _send(self, **populated: Any) -> None:
        """Send one adapter data set with the given ports populated and
        typed nulls everywhere else (the pipeline consumes all four input
        ports every step)."""
        ids = self._ads.AdapterDataSet.create()
        for port in self.INPUT_PORTS:
            if port in populated and populated[port] is not None:
                ids[port] = populated[port]
            else:
                type_name = ("uchar_vector" if port == "iqr_model" else port)
                ids.add_nullptr(port, type_name)
        self._pipeline.send(ids)

    def _receive(self) -> Dict[str, Any]:
        """Receive one adapter data set; return {port: value}."""
        ods = self._pipeline.receive()
        if ods.is_end_of_data():
            raise RuntimeError("Query pipeline terminated unexpectedly")
        out = {}
        for port, _datum in ods:
            try:
                out[port] = ods[port]
            except TypeError:
                _log(f"Ignoring unconvertible output port: {port}")
        return out

    def _capture_outputs(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Store the computed model and serialize result sets from one
        pipeline response."""
        response: Dict[str, Any] = {}
        if out.get("iqr_model") is not None and len(out["iqr_model"]):
            self._computed_model = out["iqr_model"]
            response["model_available"] = True
        results = out.get("query_result")
        if results is not None:
            serialized = [self._serialize_result(r) for r in results]
            serialized.sort(key=lambda r: -r["relevancy_score"])
            response["results"] = serialized
            if serialized and not self._query_id:
                # Auto-query fast path: adopt the backend-assigned query id
                self._query_id = serialized[0]["query_id"]
        feedback = out.get("feedback_request")
        if feedback:
            response["feedback_requests"] = [
                self._serialize_result(r) for r in feedback]
        return response

    def _serialize_result(self, qr: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "instance_id": qr.instance_id,
            "query_id": self._uid_str(qr.query_id),
            "stream_id": qr.stream_id,
            "relevancy_score": qr.relevancy_score,
            "start_frame": None,
            "end_frame": None,
            "tracks": [],
        }
        for key, ts in (("start_frame", qr.start_time()),
                        ("end_frame", qr.end_time())):
            try:
                if ts is not None and ts.has_valid_frame():
                    result[key] = ts.get_frame()
            except Exception:
                pass
        track_set = qr.tracks
        if track_set is not None:
            for track in track_set.tracks():
                states = []
                for state in track:
                    entry = {"frame": state.frame_id}
                    detection = getattr(state, "detection", None)
                    detection = detection() if callable(detection) else None
                    if detection is not None:
                        box = detection.bounding_box
                        entry["bbox"] = [box.min_x(), box.min_y(),
                                         box.max_x(), box.max_y()]
                    states.append(entry)
                result["tracks"].append({"id": track.id, "states": states})
        return result

    @staticmethod
    def _uid_str(uid: Any) -> str:
        try:
            return uid.value()
        except Exception:
            return str(uid)

    def _model_to_b64(self, model: Any) -> str:
        return base64.b64encode(bytes(bytearray(model))).decode("ascii")

    def _model_from_b64(self, model_b64: str) -> Any:
        return self._ads.VectorUChar(
            list(base64.b64decode(model_b64.encode("ascii"))))

    # ------------------------------------------------------------- commands
    def formulate_query(self, image_path: str,
                        boxes: Optional[List[List[float]]] = None
                        ) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise ValueError(f"Exemplar image not found: {image_path}")

        kvt = self._kvt
        self._formulation_count += 1
        request = kvt.DescriptorRequest()
        request.id = kvt.UID(f"DIVE-QF-{self._formulation_count}")
        request.data_location = os.path.abspath(image_path)
        if boxes:
            request.spatial_regions = [
                kvt.BoundingBoxI(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in boxes]

        # A new formulation starts a fresh query
        self._descriptors = []
        self._query_id = ""
        self._computed_model = None

        self._send(descriptor_request=request)
        out = self._receive()

        descriptors = out.get("track_descriptor_set") or []
        self._descriptors = list(descriptors)

        response = {"success": True,
                    "descriptor_count": len(self._descriptors)}
        # When boxes were provided the pipeline already ran the query
        # (auto-query fast path); pass those results straight through.
        response.update(self._capture_outputs(out))
        return response

    def process_query(self, threshold: float = 0.0,
                      iqr_model_b64: Optional[str] = None) -> Dict[str, Any]:
        if not self._descriptors:
            raise ValueError(
                "No query descriptors available; run formulate_query first")

        kvt = self._kvt
        if not self._query_id:
            self._formulation_count += 1
            self._query_id = f"DIVE-QUERY-{self._formulation_count}"

        query = kvt.DatabaseQuery()
        query.id = kvt.UID(self._query_id)
        query.type = kvt.query_type.SIMILARITY
        query.threshold = threshold
        query.descriptors = self._descriptors

        if iqr_model_b64:
            self._query_model = self._model_from_b64(iqr_model_b64)

        self._send(database_query=query, iqr_model=self._query_model)
        out = self._receive()

        response = {"success": True}
        response.update(self._capture_outputs(out))
        return response

    def refine(self, positive_ids: List[int],
               negative_ids: List[int]) -> Dict[str, Any]:
        if not positive_ids and not negative_ids:
            raise ValueError("No feedback provided")
        if self._computed_model is None and not self._query_id:
            raise ValueError("No active query to refine")

        kvt = self._kvt
        feedback = kvt.IQRFeedback()
        feedback.query_id = kvt.UID(self._query_id)
        feedback.positive_ids = [int(i) for i in positive_ids]
        feedback.negative_ids = [int(i) for i in negative_ids]

        self._send(iqr_feedback=feedback, iqr_model=self._computed_model)
        out = self._receive()

        response = {"success": True}
        response.update(self._capture_outputs(out))
        return response

    def export_model(self, output_path: Optional[str] = None
                     ) -> Dict[str, Any]:
        if self._computed_model is None or not len(self._computed_model):
            raise ValueError(
                "No trained IQR model available; refine a query first")
        model_bytes = bytes(bytearray(self._computed_model))
        if output_path:
            with open(output_path, "wb") as f:
                f.write(model_bytes)
            _log(f"Exported IQR model to {output_path}")
        return {"success": True,
                "model_b64": base64.b64encode(model_bytes).decode("ascii"),
                "output_path": output_path}

    def status(self) -> Dict[str, Any]:
        return {
            "success": True,
            "index_open": True,
            "index_dir": self.index_dir,
            "descriptor_count": len(self._descriptors),
            "query_id": self._query_id,
            "model_available": self._computed_model is not None,
        }

    def close(self) -> None:
        try:
            self._pipeline.send_end_of_input()
            # Drain remaining outputs so the pipeline can wind down
            while not self._pipeline.at_end():
                ods = self._pipeline.receive()
                if ods.is_end_of_data():
                    break
        except Exception as e:
            _log(f"Error winding down pipeline: {e}")
        try:
            self._pipeline.stop()
        except Exception:
            pass
        PostgresInstance.stop()
        _log(f"Index closed: {self.index_dir}")


# --------------------------------------------------------------------------
# Service loop
# --------------------------------------------------------------------------
class QueryService:
    def __init__(self, pipeline_file: Optional[str]):
        self._pipeline_file = pipeline_file
        self._session: Optional[QuerySession] = None

    def _resolve_pipeline_file(self, override: Optional[str]) -> str:
        candidates = [override, self._pipeline_file]
        install = os.environ.get("VIAME_INSTALL")
        if install:
            candidates.append(os.path.join(
                install, "configs", "pipelines",
                "query_retrieval_and_iqr.pipe"))
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        raise ValueError(
            "Unable to locate query_retrieval_and_iqr.pipe; pass "
            "--pipeline-file or set VIAME_INSTALL")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = request.get("command")

        if command == "open_index":
            if self._session is not None:
                self._session.close()
                self._session = None
            self._session = QuerySession(
                request["index_dir"],
                self._resolve_pipeline_file(request.get("pipeline_file")))
            return {"success": True, "index_dir": self._session.index_dir}

        if command == "close_index":
            if self._session is not None:
                self._session.close()
                self._session = None
            return {"success": True}

        if command == "status":
            if self._session is None:
                return {"success": True, "index_open": False}
            return self._session.status()

        if self._session is None:
            raise ValueError(f"No index open (required for '{command}')")

        if command == "formulate_query":
            return self._session.formulate_query(
                request["image_path"], request.get("boxes"))
        if command == "process_query":
            return self._session.process_query(
                threshold=request.get("threshold", 0.0),
                iqr_model_b64=request.get("iqr_model_b64"))
        if command == "refine":
            return self._session.refine(
                request.get("positive_ids", []),
                request.get("negative_ids", []))
        if command == "export_model":
            return self._session.export_model(request.get("output_path"))

        raise ValueError(f"Unknown command: {command}")

    def run(self) -> None:
        _log("Service started, waiting for requests...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")

                if request.get("command") == "shutdown":
                    _log("Shutdown requested")
                    if self._session is not None:
                        self._session.close()
                        self._session = None
                    self._respond({"id": request_id, "success": True,
                                   "message": "Shutting down"})
                    break

                response = self.handle_request(request)
                response["id"] = request_id
                self._respond(response)

            except json.JSONDecodeError as e:
                self._respond({"id": request_id, "success": False,
                               "error": f"Invalid JSON: {e}"})
            except Exception as e:
                _log(f"Error processing request: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._respond({"id": request_id, "success": False,
                               "error": str(e)})

        # Always release the pipeline + postgres on exit (including stdin EOF
        # when the parent process dies without sending shutdown).
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        _log("Service shutting down")

    @staticmethod
    def _respond(response: Dict[str, Any]) -> None:
        print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="VIAME video query / IQR service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline-file",
        default=None,
        help="Query pipeline to run (defaults to "
             "$VIAME_INSTALL/configs/pipelines/query_retrieval_and_iqr.pipe)",
    )
    parser.add_argument(
        "--viame-path",
        default=None,
        help="Path to VIAME install directory (sets VIAME_INSTALL env var)",
    )
    args = parser.parse_args()

    if args.viame_path:
        os.environ["VIAME_INSTALL"] = args.viame_path

    service = QueryService(args.pipeline_file)
    try:
        service.run()
    except KeyboardInterrupt:
        _log("Interrupted")
    except Exception as e:
        _log(f"Fatal error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
