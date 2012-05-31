#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


from vistk.pipeline import config
from vistk.pipeline import datum
from vistk.pipeline import edge
from vistk.pipeline import pipeline
from vistk.pipeline import process
from vistk.pipeline import schedule
from vistk.pipeline import utils

import threading


class UnsupportedProcess(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        fmt = "The process '%s' does not support running in a Python thread"
        return (fmt % self.name)


class PyThreadPerProcessSchedule(schedule.PythonSchedule):
    """ Runs each process in a pipeline in its own thread.
    """

    def __init__(self, pipe, conf):
        schedule.PythonSchedule.__init__(self, pipe, conf)

        p = self.pipeline()
        names = p.process_names()

        no_threads = process.PythonProcess.constraint_no_threads

        for name in names:
            proc = p.process_by_name(name)
            constraints = proc.constraints()

            if no_threads in constraints:
                raise UnsupportedProcess(name)

        self.threads = []
        self.event = threading.Event()
        self._make_monitor_edge_config()

    def _start(self):
        p = self.pipeline()
        names = p.process_names()

        for name in names:
            proc = p.process_by_name(name)

            thread = threading.Thread(target=self._run_process, name=name, args=(proc,))

            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def _wait(self):
        for thread in self.threads:
            thread.join()

    def _stop(self):
        self.event.set()

    def _run_process(self, proc):
        utils.name_thread(proc.name())

        monitor = edge.Edge(self.edge_conf)

        proc.connect_output_port(process.PythonProcess.port_heartbeat, monitor)

        complete = False

        while not complete and not self.event.is_set():
            proc.step()

            while monitor.has_data():
                edat = monitor.get_datum()
                dat = edat.datum

                if dat.type() == datum.DatumType.complete:
                    complete = True

    def _make_monitor_edge_config(self):
        self.edge_conf = config.empty_config()
