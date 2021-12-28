from enum import auto, Enum


class SQLBackendTypes(Enum):
    shortname = auto()
    ip_addr = auto()
    sqlite = auto()
