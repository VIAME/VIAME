import logging
from girder_worker import GirderWorkerPluginABC

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class ViamePlugin(GirderWorkerPluginABC):
    def __init__(self, app, *args, **kwargs):
        self.app = app

    def task_imports(self):
        # Return a list of python importable paths to the
        # plugin's path directory
        return ["viame_tasks.tasks"]
