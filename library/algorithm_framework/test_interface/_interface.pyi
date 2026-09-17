from viame.plugins import Pluggable

class Say (Pluggable):
    def says(self) -> str: ...
