#ckwg +28
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


from kwiver.vital.config import config
from kwiver.sprokit.pipeline import datum
from kwiver.sprokit.pipeline import edge
from kwiver.sprokit.pipeline import pipeline
from kwiver.sprokit.pipeline import process
from kwiver.sprokit.pipeline import scheduler
from kwiver.sprokit.pipeline import utils

import threading


class UnsupportedProcess(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        fmt = "The process '%s' does not support running in a Python thread"
        return (fmt % self.name)


class PyThreadPerProcessScheduler(scheduler.PythonScheduler):
    """ Runs each process in a pipeline in its own thread.
    """

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
        self._make_monitor_edge_config()

    def _start(self):
        p = self.pipeline()
        names = p.process_names()

        for name in names:
            proc = p.process_by_name(name)

            thread = threading.Thread(target=self._run_process, name=name, args=(proc,))

            self._threads.append(thread)

        for thread in self._threads:
            thread.start()

    def _wait(self):
        for thread in self._threads:
            thread.join()

    def _pause(self):
        self._pause_event.set()

    def _resume(self):
        self._pause_event.clear()

    def _stop(self):
        self._event.set()
        self.shutdown()

    def _run_process(self, proc):
        utils.name_thread(proc.name())

        monitor = edge.Edge(self._edge_conf)

        proc.connect_output_port(process.PythonProcess.port_heartbeat, monitor)

        complete = False

        while not complete and not self._event.is_set():
            while self._pause_event.is_set():
                self._pause_event.wait()

            proc.step()

            while monitor.has_data():
                edat = monitor.get_datum()
                dat = edat.datum

                if dat.type() == datum.DatumType.complete:
                    complete = True

    def _make_monitor_edge_config(self):
        self._edge_conf = config.empty_config()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import scheduler_factory

    module_name = 'python:schedulers'

    if scheduler_factory.is_scheduler_module_loaded(module_name):
        return

    scheduler_factory.add_scheduler('pythread_per_process',
                                    'Run each process in its own Python thread',
                                    PyThreadPerProcessScheduler)

    scheduler_factory.mark_scheduler_module_as_loaded(module_name)
