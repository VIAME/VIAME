from kwiver.vital.plugins import Pluggable

class Say (Pluggable):
    def says(self) -> str: ...
