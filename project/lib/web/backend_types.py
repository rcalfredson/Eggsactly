from enum import auto, Enum


class BackendTypes(Enum):
    sql = auto()
    filesystem = auto()
