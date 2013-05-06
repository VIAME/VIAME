#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from sprokit.pipeline import config
from sprokit.pipeline import datum
from sprokit.pipeline import edge
from sprokit.pipeline import pipeline
from sprokit.pipeline import process
from sprokit.pipeline import scheduler
from sprokit.pipeline import utils

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
