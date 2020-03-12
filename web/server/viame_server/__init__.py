import sys
from pathlib import Path
from os import getenv

from girder import events, plugin

from .client_webroot import ClientWebroot
from .viame import Viame
from .viame_detection import ViameDetection
from .utils import check_existing_annotations


env_pipelines_path = getenv("VIAME_PIPELINES_PATH")


def load_pipelines():
    if env_pipelines_path is None:
        print(
            "No pipeline path specified. "
            "Please set the VIAME_PIPELINES_PATH environment variable.",
            file=sys.stderr,
        )
        return []

    pipeline_path = Path(env_pipelines_path)
    if not pipeline_path.exists():
        print("Specified pipeline path does not exist!", file=sys.stderr)
        return []

    return [path.name for path in pipeline_path.glob("./*.pipe")]


class GirderPlugin(plugin.GirderPlugin):
    def load(self, info):
        info["apiRoot"].viame = Viame(pipelines=load_pipelines())
        info["apiRoot"].viame_detection = ViameDetection()
        # Relocate Girder
        info["serverRoot"], info["serverRoot"].girder = (
            ClientWebroot(),
            info["serverRoot"],
        )
        info["serverRoot"].api = info["serverRoot"].girder.api

        events.bind(
            "filesystem_assetstore_imported",
            "check_annotations",
            check_existing_annotations,
        )
