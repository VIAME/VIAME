from typing import Type
from typing import TypeVar

from kwiver.vital.config import Config
from kwiver.vital import plugin_management

from . import Say


T = TypeVar("T", bound="PythonTheyImpl")


class PythonTheyImpl(Say):

    def says(self) -> str:
        return "In Python they say " + self.speaker.says()

    @classmethod
    def from_config(cls: Type[T], c: Config) -> T:
        retVal = PythonTheyImpl()
        retVal.speaker = plugin_management.SayFactory().create(
            c.get_value("speaker", "PythonImpl"), c
        )

        return retVal

    @classmethod
    def get_default_config(cls, c: Config): ...  # nothing to set
