# ckwg +28
# Copyright 2012 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from kwiver.vital.config import empty_config
from kwiver.sprokit.pipeline import datum
from kwiver.sprokit.pipeline import edge
from kwiver.sprokit.pipeline import pipeline
from kwiver.sprokit.pipeline import process
from kwiver.sprokit.pipeline import scheduler
from kwiver.sprokit.pipeline import utils

import sys
import threading
import traceback


class UnsupportedProcess(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        fmt = "The process '%s' does not support running in a Python thread"
        return fmt % self.name


class ProcessException(Exception):
    """Exception raised when a process encounters an error during execution."""

    def __init__(self, process_name, original_exception, tb_str):
        self.process_name = process_name
        self.original_exception = original_exception
        self.tb_str = tb_str

    def __str__(self):
        return "Process '{}' raised an exception:\n{}".format(
            self.process_name, self.tb_str
        )


class PyThreadPerProcessScheduler(scheduler.PythonScheduler):
    """Runs each process in a pipeline in its own thread."""

    def __init__(self, pipe, conf):
        scheduler.PythonScheduler.__init__(self, pipe, conf)

        p = self.pipeline()
        names = p.process_names()

        no_threads = process.PythonProcess.property_no_threads

        for name in names:
            proc = p.process_by_name(name)
            properties = proc.properties()

            if no_threads in properties:
                raise UnsupportedProcess(name)

        self._threads = []
        self._pause_event = threading.Event()
        self._event = threading.Event()
        self._error_lock = threading.Lock()
        self._process_exception = None
        self._make_monitor_edge_config()

    def _start(self):
        p = self.pipeline()
        names = p.process_names()

        for name in names:
            proc = p.process_by_name(name)

            thread = threading.Thread(target=self._run_process, name=name, args=(proc,))
            # A process blocked reading an edge cannot observe the stop event.
            # Daemonising means such a thread can never keep the interpreter
            # alive once the pipeline has failed.
            thread.daemon = True

            self._threads.append(thread)

        for thread in self._threads:
            thread.start()

    def _wait(self):
        # Do not join unconditionally. When a process raises, the threads that
        # feed or drain it are typically blocked inside a step() on an edge and
        # will never look at the stop event, so joining them hangs the pipeline
        # instead of reporting the failure.
        while True:
            alive = [t for t in self._threads if t.is_alive()]

            if not alive:
                break

            with self._error_lock:
                failed = self._process_exception is not None

            if failed:
                break

            for thread in alive:
                thread.join(0.1)

        if self._process_exception is not None:
            # Grace period for threads that can still unwind on their own.
            # Any that remain are blocked on an edge whose peer died and are
            # daemonic, so they cannot be unblocked from here; abandon them.
            for thread in self._threads:
                if thread.is_alive():
                    thread.join(0.5)

        # Re-raise any exception that occurred in a process thread
        if self._process_exception is not None:
            raise self._process_exception

    def _pause(self):
        self._pause_event.set()

    def _resume(self):
        self._pause_event.clear()

    def _stop(self):
        self._event.set()
        # Do not call self.shutdown() here. The C++ scheduler::stop() already
        # holds the non-recursive scheduler mutex when it calls _stop(), and
        # shutdown() takes that same mutex, so calling it here deadlocks. The
        # base class performs the shutdown once _stop() returns.

    def _run_process(self, proc):
        utils.name_thread(proc.name())

        monitor = edge.Edge(self._edge_conf)

        proc.connect_output_port(process.PythonProcess.port_heartbeat, monitor)

        complete = False

        try:
            while not complete and not self._event.is_set():
                while self._pause_event.is_set():
                    self._pause_event.wait()

                proc.step()

                while monitor.has_data():
                    edat = monitor.get_datum()
                    dat = edat.datum

                    if dat.type() == datum.DatumType.complete:
                        complete = True
        except Exception as e:
            # Capture the exception details
            tb_str = traceback.format_exc()

            # Store the exception (thread-safe)
            with self._error_lock:
                if self._process_exception is None:
                    self._process_exception = ProcessException(proc.name(), e, tb_str)

            # Print the error immediately so users see what went wrong
            print("Error in process '{}':".format(proc.name()), file=sys.stderr)
            print(tb_str, file=sys.stderr)

            # Signal all threads to stop
            self._event.set()

    def _make_monitor_edge_config(self):
        self._edge_conf = empty_config()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import scheduler_factory

    module_name = "python:schedulers"

    if scheduler_factory.is_scheduler_module_loaded(module_name):
        return

    scheduler_factory.add_scheduler(
        "pythread_per_process",
        "Run each process in its own Python thread",
        PyThreadPerProcessScheduler,
    )

    scheduler_factory.mark_scheduler_module_as_loaded(module_name)
