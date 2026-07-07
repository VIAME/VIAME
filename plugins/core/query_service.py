#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Video Query / IQR Service

Hosts VIAME's video search and iterative query refinement (IQR) engine as a
persistent service for GUI frontends (DIVE desktop). Wraps the
``query_retrieval_and_iqr.pipe`` KWIVER pipeline in embedded pipelines and
manages the per-index embedded PostgreSQL instances, mirroring the protocol
the VIQUI (vivia) KIP query session speaks:

  input adapter ports : descriptor_request, database_query, iqr_feedback,
                        iqr_model  (exactly one populated per step; the rest
                        carry typed null sptrs)
  output adapter ports: track_descriptor_set, query_result, feedback_request,
                        iqr_model

An "index" is a directory containing the ``database/`` folder produced by
``process_video.py --build-index`` (embedded PostgreSQL data at
``database/SQL``, ITQ/LSH files at ``database/ITQ``).

FEDERATED SEARCH: multiple indexes may be opened at once. The first becomes
the primary session (default postgres port, full pipeline including the
exemplar descriptor path); the rest are secondary sessions running a reduced
query-only pipeline against their own postgres instance on an incremented
port. A query formulates exemplar descriptors once on the primary and fans
the similarity query out to every session; results are merged by relevancy.
Refinement feedback routes to the session that owns each result; sessions
without any accumulated feedback are re-scored with the canonical model (the
model from the most-adjudicated session), keeping scores comparable.

Protocol: newline-delimited JSON requests on stdin, JSON responses on stdout,
each response echoing the request ``id``. Commands:

  open_index      {index_dir} | {index_dirs: [dir, ...]}
  close_index     {}
  remove_streams  {index_dir, streams: [video_name, ...]}
      Deletes every database row belonging to the given streams (e.g. when
      a video is removed from the index). Stale ITQ/LSH hash entries are
      tolerated by the query engine and disappear on the next index build.
  status          {}
  formulate_query {image_path, boxes?: [[x1,y1,x2,y2],...]}
  process_query   {threshold?, iqr_model_b64?}
  refine          {positive_ids: [ref], negative_ids: [ref]}
      refs are "<session>:<instance_id>" strings (bare ints refer to
      session 0 for single-index compatibility)
  export_model    {output_path?}
  shutdown        {}

Result entries are serialized as
  {ref, session, index_dir, instance_id, stream_id, relevancy_score,
   start_frame, end_frame, tracks: [{id, states: [{frame, bbox}]}]}

Usage:
    python -m viame.core.query_service [--pipeline-file <query pipe>]
"""

import argparse
import base64
import faulthandler
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Native pipeline code runs in-process; make hard crashes debuggable.
faulthandler.enable(file=sys.stderr)

BASE_DB_PORT = 5432
MERGED_RESULT_LIMIT = 200


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
# instances belonging to its own index directories.
# --------------------------------------------------------------------------
def _port_is_free(port: int) -> bool:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


class PostgresInstance:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.port: Optional[int] = None
        # True when an already-running server was adopted rather than started
        self.adopted = False
        self.sql_dir = os.path.join(index_dir, "database", "SQL")
        self.log_file = os.path.join(index_dir, "database", "SQL_Log_File")

    def _pg_ctl(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [_exe("pg_ctl"), "-D", self.sql_dir] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def is_running(self) -> bool:
        return self._pg_ctl(["status"]).returncode == 0

    def _running_port(self) -> Optional[int]:
        """Port of the live server for this data directory, if any (line 4
        of postmaster.pid)."""
        if not self.is_running():
            return None
        pid_file = os.path.join(self.sql_dir, "postmaster.pid")
        try:
            with open(pid_file) as f:
                lines = f.read().splitlines()
            return int(lines[3])
        except Exception:
            return None

    def ensure_started(self, avoid_ports: List[int]) -> int:
        """Start (or adopt) the postgres instance for this index and return
        the port it listens on. An instance left running by a previous index
        build or session is adopted at whatever port it already uses;
        otherwise the first free port not in ``avoid_ports`` is chosen."""
        running = self._running_port()
        if running is not None:
            _log(f"Adopting running postgres for {self.index_dir} "
                 f"on port {running}")
            self.port = running
            self.adopted = True
            return running

        port = BASE_DB_PORT
        while port in avoid_ports or not _port_is_free(port):
            port += 1
        self.port = port

        start_args = ["-w", "-t", "20", "-l", self.log_file,
                      "-o", f"-p {port}", "start"]
        # Recover from a previous unclean shutdown (stale postmaster.pid with
        # no live server behind it); pg_ctl start handles most cases itself,
        # but a pid file pointing at a recycled pid can block startup.
        pid_file = os.path.join(self.sql_dir, "postmaster.pid")
        result = self._pg_ctl(start_args)
        if result.returncode != 0 and os.path.exists(pid_file):
            _log("postgres start failed; removing stale postmaster.pid "
                 "and retrying")
            os.remove(pid_file)
            result = self._pg_ctl(start_args)
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to start postgres for {self.index_dir}: "
                f"{result.stdout}")
        return port

    def stop(self) -> None:
        self._pg_ctl(["-m", "fast", "stop"])
        self._wait_for_port_available()

    def _wait_for_port_available(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.port is None or _port_is_free(self.port):
                return True
            time.sleep(0.5)
        return False


# Removes every row belonging to one stream (video/sequence identifier).
# Child tables (keyed by UID only) go first. The descriptor vectors and
# track geometry all key on VIDEO_NAME.
REMOVE_STREAM_SQL = (
    "DELETE FROM TRACK_DESCRIPTOR_TRACK WHERE UID IN "
    "(SELECT UID FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream});\n"
    "DELETE FROM TRACK_DESCRIPTOR_HISTORY WHERE UID IN "
    "(SELECT UID FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream});\n"
    "DELETE FROM TRACK_DESCRIPTOR WHERE VIDEO_NAME = {stream};\n"
    "DELETE FROM DESCRIPTOR WHERE VIDEO_NAME = {stream};\n"
    "DELETE FROM OBJECT_TRACK WHERE VIDEO_NAME = {stream};"
)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# --------------------------------------------------------------------------
# Pipe templating for secondary sessions
# --------------------------------------------------------------------------
DEFAULT_CONN_STR = "postgresql:host=localhost;user=postgres"

SECONDARY_OUTER_PIPE = """# Generated by viame.core.query_service -- do not edit.
# Reduced query pipeline for a secondary (federated) index session: no
# exemplar descriptor path (formulation happens on the primary session).

config _pipeline:_edge
  :capacity                                    10

process in_adapt
 :: input_adapter

process out_adapt
 :: output_adapter

process database_query_handler
  :: perform_query
  :external_handler                             true
  :external_pipeline_file                       {inner_pipe}
  :database_folder                              {database_folder}
  :max_result_count                             200
  :use_tracks_for_history                       true
  :merge_duplicate_results                      true
  :unused_descriptors_as_negative               false
  :descriptor_query:type                        db
  :descriptor_query:db:conn_str                 {conn_str}

connect from in_adapt.database_query
        to   database_query_handler.database_query
connect from in_adapt.iqr_feedback
        to   database_query_handler.iqr_feedback
connect from in_adapt.iqr_model
        to   database_query_handler.iqr_model

connect from database_query_handler.query_result
        to   out_adapt.query_result
connect from database_query_handler.feedback_request
        to   out_adapt.feedback_request
connect from database_query_handler.iqr_model
        to   out_adapt.iqr_model
"""


def _write_session_pipes(index_dir: str, port: int,
                         installed_pipe: str, primary: bool) -> str:
    """Generate CWD-independent pipeline files for a session: an outer pipe
    plus a copy of the inner query_and_iqr.pipe, all with absolute index
    paths and the session's postgres port. The primary session gets the full
    installed pipeline (including the exemplar descriptor path); secondaries
    get a reduced query-only pipeline so heavy formulation models load once.
    Returns the outer pipe path."""
    session_dir = os.path.join(index_dir, ".session")
    os.makedirs(session_dir, exist_ok=True)

    installed_dir = os.path.dirname(installed_pipe)
    database_folder = os.path.join(index_dir, "database")
    conn_str = f"{DEFAULT_CONN_STR};port={port}"

    # Inner pipe: absolute database folder + session port
    inner_src = os.path.join(installed_dir, "query_and_iqr.pipe")
    with open(inner_src) as f:
        inner = f.read()
    inner = inner.replace(DEFAULT_CONN_STR, conn_str)
    inner = inner.replace(
        ":database_folder                             database",
        f":database_folder                             {database_folder}")
    inner_path = os.path.join(session_dir, "query_and_iqr.pipe")
    with open(inner_path, "w") as f:
        f.write(inner)

    outer_path = os.path.join(session_dir, "query_retrieval.pipe")
    if primary:
        # Full installed pipeline, rewritten to be CWD/port independent
        with open(installed_pipe) as f:
            outer = f.read()
        outer = outer.replace(DEFAULT_CONN_STR, conn_str)
        outer = outer.replace(
            ":database_folder                             database",
            f":database_folder                             {database_folder}")
        outer = outer.replace(
            ":query_folder                                database/Queries",
            f":query_folder                                {database_folder}/Queries")
        outer = outer.replace(
            "relativepath image_pipeline_file =           "
            "query_image_exemplar.pipe",
            ":image_pipeline_file                         "
            f"{os.path.join(installed_dir, 'query_image_exemplar.pipe')}")
        outer = outer.replace(
            "relativepath external_pipeline_file =         query_and_iqr.pipe",
            f":external_pipeline_file                       {inner_path}")
        with open(outer_path, "w") as f:
            f.write(outer)
    else:
        with open(outer_path, "w") as f:
            f.write(SECONDARY_OUTER_PIPE.format(
                inner_pipe=inner_path,
                database_folder=database_folder,
                conn_str=conn_str,
            ))
    return outer_path


# --------------------------------------------------------------------------
# Query session: one open index + embedded pipeline + evolving IQR model
# --------------------------------------------------------------------------
class QuerySession:
    """Wraps one index directory's embedded query pipeline."""

    def __init__(self, index_dir: str, primary: bool,
                 pipeline_file: str, avoid_ports: List[int]):
        # Import lazily so --help etc. work without a full VIAME environment.
        from kwiver.sprokit.adapters import adapter_data_set, embedded_pipeline
        from kwiver.vital import types as kvt

        self._ads = adapter_data_set
        self._kvt = kvt

        self.index_dir = os.path.abspath(index_dir)
        self.primary = primary
        self.computed_model: Optional[Any] = None  # backend-owned VectorUChar
        self.query_id: str = ""
        # Cumulative feedback (refs resolved to plain instance ids)
        self.cumulative_positive: List[int] = []
        self.cumulative_negative: List[int] = []

        if not os.path.isdir(
                os.path.join(self.index_dir, "database", "ITQ")):
            raise ValueError(
                f"{index_dir} does not contain a built search index "
                "(missing database/ITQ)")

        self._postgres = PostgresInstance(self.index_dir)
        self.port = self._postgres.ensure_started(avoid_ports)

        pipe_path = _write_session_pipes(
            self.index_dir, self.port, pipeline_file, primary)
        pipe_dir = os.path.dirname(pipe_path)

        _log(f"Building query pipeline for {self.index_dir} "
             f"(port {self.port}, {'primary' if primary else 'secondary'})")
        self._pipeline = embedded_pipeline.EmbeddedPipeline()
        # def_dir anchors `relativepath` config entries (e.g. the inner
        # query_and_iqr.pipe) to the pipe file rather than the CWD.
        self._pipeline.build_pipeline(pipe_path, pipe_dir)
        self._pipeline.start()
        self._input_ports = set(self._pipeline.input_port_names())
        _log(f"Index opened: {self.index_dir}")

    # -------------------------------------------------------------- helpers
    def _send(self, **populated: Any) -> None:
        """Send one adapter data set populating the given ports and typed
        nulls on every other input port the pipeline exposes."""
        ids = self._ads.AdapterDataSet.create()
        for port in self._input_ports:
            if port in populated and populated[port] is not None:
                ids[port] = populated[port]
            else:
                type_name = ("uchar_vector" if port == "iqr_model" else port)
                ids.add_nullptr(port, type_name)  # port names match type names
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

    def capture_outputs(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Store the computed model and return raw result lists."""
        response: Dict[str, Any] = {}
        if out.get("iqr_model") is not None and len(out["iqr_model"]):
            self.computed_model = out["iqr_model"]
            response["model_available"] = True
        if out.get("query_result") is not None:
            response["results"] = list(out["query_result"])
            if response["results"] and not self.query_id:
                # Auto-query fast path: adopt the backend-assigned query id
                self.query_id = _uid_str(response["results"][0].query_id)
        if out.get("feedback_request"):
            response["feedback_requests"] = list(out["feedback_request"])
        return response

    def make_query(self, descriptors: List[Any], threshold: float) -> Any:
        kvt = self._kvt
        if not self.query_id:
            self.query_id = f"DIVE-QUERY-{os.path.basename(self.index_dir)}"
        query = kvt.DatabaseQuery()
        query.id = kvt.UID(self.query_id)
        query.type = kvt.query_type.SIMILARITY
        query.threshold = threshold
        query.descriptors = descriptors
        return query

    # ------------------------------------------------------------- commands
    def formulate(self, image_path: str,
                  boxes: Optional[List[List[float]]]) -> Dict[str, Any]:
        """Primary-session only: compute exemplar descriptors (and, when
        boxes are given, the pipeline auto-runs the first query)."""
        kvt = self._kvt
        request = kvt.DescriptorRequest()
        request.id = kvt.UID("DIVE-QF")
        request.data_location = os.path.abspath(image_path)
        if boxes:
            request.spatial_regions = [
                kvt.BoundingBoxI(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in boxes]
        self._send(descriptor_request=request)
        out = self._receive()
        response = self.capture_outputs(out)
        response["descriptors"] = list(out.get("track_descriptor_set") or [])
        return response

    def process_query(self, descriptors: List[Any], threshold: float,
                      model: Optional[Any]) -> Dict[str, Any]:
        query = self.make_query(descriptors, threshold)
        self._send(database_query=query, iqr_model=model)
        return self.capture_outputs(self._receive())

    def refine(self, positive_ids: List[int],
               negative_ids: List[int]) -> Dict[str, Any]:
        kvt = self._kvt
        feedback = kvt.IQRFeedback()
        feedback.query_id = kvt.UID(self.query_id)
        feedback.positive_ids = [int(i) for i in positive_ids]
        feedback.negative_ids = [int(i) for i in negative_ids]
        self._send(iqr_feedback=feedback, iqr_model=self.computed_model)
        return self.capture_outputs(self._receive())

    @property
    def feedback_count(self) -> int:
        return len(self.cumulative_positive) + len(self.cumulative_negative)

    def reset_query_state(self) -> None:
        self.computed_model = None
        self.query_id = ""
        self.cumulative_positive = []
        self.cumulative_negative = []

    def close(self) -> None:
        try:
            self._pipeline.send_end_of_input()
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
        self._postgres.stop()
        _log(f"Index closed: {self.index_dir}")


def _uid_str(uid: Any) -> str:
    try:
        return uid.value()
    except Exception:
        return str(uid)


def _serialize_result(qr: Any, session: int, index_dir: str,
                      relevancy: Optional[float] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ref": f"{session}:{qr.instance_id}",
        "session": session,
        "index_dir": index_dir,
        "instance_id": qr.instance_id,
        "query_id": _uid_str(qr.query_id),
        "stream_id": qr.stream_id,
        "relevancy_score": (relevancy if relevancy is not None
                            else qr.relevancy_score),
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


def _parse_ref(ref: Any) -> Tuple[int, int]:
    """Parse a result reference: "<session>:<instance_id>" or a bare
    instance id (session 0, single-index compatibility)."""
    if isinstance(ref, str) and ":" in ref:
        session, instance = ref.split(":", 1)
        return int(session), int(instance)
    return 0, int(ref)


# --------------------------------------------------------------------------
# Service loop
# --------------------------------------------------------------------------
class QueryService:
    def __init__(self, pipeline_file: Optional[str],
                 protocol_out: Optional[Any] = None):
        self._pipeline_file = pipeline_file
        self._sessions: List[QuerySession] = []
        self._descriptors: List[Any] = []
        self._protocol_out = protocol_out if protocol_out is not None else sys.stdout

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

    # ----------------------------------------------------------- federation
    def _close_all(self) -> None:
        for session in reversed(self._sessions):
            try:
                session.close()
            except Exception as e:
                _log(f"Error closing {session.index_dir}: {e}")
        self._sessions = []
        self._descriptors = []

    def _open(self, index_dirs: List[str],
              pipeline_override: Optional[str]) -> Dict[str, Any]:
        self._close_all()
        pipeline_file = self._resolve_pipeline_file(pipeline_override)
        opened: List[str] = []
        try:
            for i, index_dir in enumerate(index_dirs):
                avoid_ports = [s.port for s in self._sessions]
                self._sessions.append(QuerySession(
                    index_dir, primary=(i == 0),
                    pipeline_file=pipeline_file, avoid_ports=avoid_ports))
                opened.append(os.path.abspath(index_dir))
        except Exception:
            self._close_all()
            raise
        return {"success": True, "index_dirs": opened}

    def _merge(self, per_session: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge per-session raw outputs into one ranked response."""
        results: List[Dict[str, Any]] = []
        feedback: List[Dict[str, Any]] = []
        model_available = False
        for i, response in enumerate(per_session):
            if response is None:
                continue
            index_dir = self._sessions[i].index_dir
            for qr in response.get("results", []):
                results.append(_serialize_result(qr, i, index_dir))
            for qr in response.get("feedback_requests", []):
                feedback.append(_serialize_result(qr, i, index_dir))
            if response.get("model_available"):
                model_available = True
        results.sort(key=lambda r: -r["relevancy_score"])
        merged: Dict[str, Any] = {
            "success": True,
            "results": results[:MERGED_RESULT_LIMIT],
        }
        if feedback:
            merged["feedback_requests"] = feedback
        if model_available:
            merged["model_available"] = True
        return merged

    def _canonical_session(self) -> Optional[QuerySession]:
        """The session whose model is broadcast to feedback-less sessions:
        the one with the most accumulated adjudications and a model."""
        best = None
        for session in self._sessions:
            if session.computed_model is None:
                continue
            if best is None or session.feedback_count > best.feedback_count:
                best = session
        return best

    # ------------------------------------------------------------- commands
    def _formulate_query(self, image_path: str,
                         boxes: Optional[List[List[float]]]) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise ValueError(f"Exemplar image not found: {image_path}")

        for session in self._sessions:
            session.reset_query_state()

        primary = self._sessions[0]
        primary_out = primary.formulate(image_path, boxes)
        self._descriptors = primary_out.pop("descriptors")

        per_session: List[Optional[Dict[str, Any]]] = [primary_out]
        # The primary auto-ran the query when boxes were provided; if not,
        # every session (primary included) runs it explicitly below.
        if "results" not in primary_out:
            per_session[0] = primary.process_query(self._descriptors, 0.0, None)
        for session in self._sessions[1:]:
            per_session.append(
                session.process_query(self._descriptors, 0.0, None))

        merged = self._merge(per_session)
        merged["descriptor_count"] = len(self._descriptors)
        return merged

    def _process_query(self, threshold: float,
                       iqr_model_b64: Optional[str]) -> Dict[str, Any]:
        if not self._descriptors:
            raise ValueError(
                "No query descriptors available; run formulate_query first")
        model = None
        if iqr_model_b64:
            model = self._model_from_b64(iqr_model_b64)
        per_session = [
            session.process_query(self._descriptors, threshold, model)
            for session in self._sessions]
        return self._merge(per_session)

    def _refine(self, positive_refs: List[Any],
                negative_refs: List[Any]) -> Dict[str, Any]:
        if not positive_refs and not negative_refs:
            raise ValueError("No feedback provided")

        positives: Dict[int, List[int]] = {}
        negatives: Dict[int, List[int]] = {}
        for ref in positive_refs:
            session, instance = _parse_ref(ref)
            positives.setdefault(session, []).append(instance)
        for ref in negative_refs:
            session, instance = _parse_ref(ref)
            negatives.setdefault(session, []).append(instance)

        for idx in list(positives) + list(negatives):
            if idx < 0 or idx >= len(self._sessions):
                raise ValueError(f"Unknown session in feedback ref: {idx}")

        per_session: List[Optional[Dict[str, Any]]] = [None] * len(self._sessions)

        # Sessions receiving feedback (new this round, or resent cumulative
        # so re-ranking stays consistent) refine with their own model.
        for i, session in enumerate(self._sessions):
            new_pos = positives.get(i, [])
            new_neg = negatives.get(i, [])
            session.cumulative_positive.extend(
                p for p in new_pos if p not in session.cumulative_positive)
            session.cumulative_negative.extend(
                n for n in new_neg if n not in session.cumulative_negative)
            if session.feedback_count > 0:
                per_session[i] = session.refine(
                    session.cumulative_positive, session.cumulative_negative)

        # Sessions with no feedback at all are re-scored with the canonical
        # model so their results stay comparable with adjudicated sessions.
        canonical = self._canonical_session()
        for i, session in enumerate(self._sessions):
            if per_session[i] is None:
                model = canonical.computed_model if canonical else None
                per_session[i] = session.process_query(
                    self._descriptors, 0.0, model)

        return self._merge(per_session)

    def _export_model(self, output_path: Optional[str]) -> Dict[str, Any]:
        canonical = self._canonical_session()
        if canonical is None or canonical.computed_model is None:
            raise ValueError(
                "No trained IQR model available; refine a query first")
        model_bytes = bytes(bytearray(canonical.computed_model))
        if output_path:
            with open(output_path, "wb") as f:
                f.write(model_bytes)
            _log(f"Exported IQR model to {output_path}")
        return {"success": True,
                "model_b64": base64.b64encode(model_bytes).decode("ascii"),
                "output_path": output_path}

    def _model_from_b64(self, model_b64: str) -> Any:
        from kwiver.sprokit.adapters import adapter_data_set
        return adapter_data_set.VectorUChar(
            list(base64.b64decode(model_b64.encode("ascii"))))

    def _remove_streams(self, index_dir: str,
                        streams: List[str]) -> Dict[str, Any]:
        """Delete all database rows for the given stream identifiers. Runs
        against the open session's postgres when the index is open (the
        session set is then closed, since in-memory descriptor indexes still
        hold the removed vectors); otherwise a temporary postgres instance
        is started (or adopted) and stopped again."""
        index_dir = os.path.abspath(index_dir)
        if not streams:
            return {"success": True, "closed_session": False}

        open_session = next(
            (s for s in self._sessions if s.index_dir == index_dir), None)
        temp_pg: Optional[PostgresInstance] = None
        if open_session is not None:
            port = open_session.port
        else:
            temp_pg = PostgresInstance(index_dir)
            port = temp_pg.ensure_started(
                [s.port for s in self._sessions])

        sql = "\n".join(
            REMOVE_STREAM_SQL.format(stream=_sql_literal(s)) for s in streams)
        try:
            result = subprocess.run(
                [_exe("psql"), "-h", "localhost", "-p", str(port),
                 "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-c", sql],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Stream removal failed: {result.stdout}")
        finally:
            # Only stop a server this call started itself
            if temp_pg is not None and not temp_pg.adopted:
                temp_pg.stop()

        closed = False
        if open_session is not None:
            # The open sessions cached descriptors in memory; close so the
            # next query reopens against the pruned database.
            self._close_all()
            closed = True
        _log(f"Removed streams {streams} from {index_dir}")
        return {"success": True, "closed_session": closed}

    def _status(self) -> Dict[str, Any]:
        return {
            "success": True,
            "index_open": bool(self._sessions),
            "index_dirs": [s.index_dir for s in self._sessions],
            "descriptor_count": len(self._descriptors),
            "model_available": any(
                s.computed_model is not None for s in self._sessions),
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = request.get("command")

        if command == "open_index":
            dirs = request.get("index_dirs")
            if not dirs:
                dirs = [request["index_dir"]]
            return self._open(dirs, request.get("pipeline_file"))

        if command == "close_index":
            self._close_all()
            return {"success": True}

        if command == "remove_streams":
            return self._remove_streams(
                request["index_dir"], request["streams"])

        if command == "status":
            return self._status()

        if not self._sessions:
            raise ValueError(f"No index open (required for '{command}')")

        if command == "formulate_query":
            return self._formulate_query(
                request["image_path"], request.get("boxes"))
        if command == "process_query":
            return self._process_query(
                threshold=request.get("threshold", 0.0),
                iqr_model_b64=request.get("iqr_model_b64"))
        if command == "refine":
            return self._refine(
                request.get("positive_ids", []),
                request.get("negative_ids", []))
        if command == "export_model":
            return self._export_model(request.get("output_path"))

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
                    self._close_all()
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

        # Always release the pipelines + postgres on exit (including stdin
        # EOF when the parent process dies without sending shutdown).
        try:
            self._close_all()
        except Exception:
            pass
        _log("Service shutting down")

    def _respond(self, response: Dict[str, Any]) -> None:
        self._protocol_out.write(json.dumps(response) + "\n")
        self._protocol_out.flush()


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

    # Reserve the real stdout for the NDJSON protocol and route every other
    # stdout writer -- including C/C++ pipeline code and plugin loaders that
    # print below the Python layer -- to stderr, so stray output can never
    # corrupt the protocol stream.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    service = QueryService(args.pipeline_file, protocol_out)
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
