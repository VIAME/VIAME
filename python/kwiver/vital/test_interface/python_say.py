from typing import Type
from typing import TypeVar

from kwiver.vital.config import Config

from . import Say


T = TypeVar("T", bound="PythonImpl")


class PythonImpl(Say):
    """Python implementation."""

    def says(self) -> str:
        return "I'm the Python implementation"

    @classmethod
    def from_config(cls: Type[T], c: Config) -> T:
        return PythonImpl()

    @classmethod
    def get_default_config(cls, c: Config): ...  # nothing to set
