import importlib.resources
from typing import IO

# Define character type IDs
CHAR_TYPE_ZH = 0
CHAR_TYPE_NUM = 1
CHAR_TYPE_ALPHA = 2
CHAR_TYPE_OTHER = 3

# Pre-defined ranges for character types
CHAR_TYPE_RANGES = [
    ((0x4E00, 0x9FA5), CHAR_TYPE_ZH),
    ((0x0030, 0x0039), CHAR_TYPE_NUM),
    ((0x0041, 0x005A), CHAR_TYPE_ALPHA),
    ((0x0061, 0x007A), CHAR_TYPE_ALPHA),
]

_MAX_CHAR_CODE = 0x9FA5 + 1
_CHAR_TYPE_LOOKUP = [CHAR_TYPE_OTHER] * _MAX_CHAR_CODE

for (start, end), char_type_id in CHAR_TYPE_RANGES:
    for char_code in range(start, end + 1):
        if char_code < _MAX_CHAR_CODE:
            _CHAR_TYPE_LOOKUP[char_code] = char_type_id


def get_module_res(module: str, name: str) -> IO[bytes]:
    return importlib.resources.files(module).joinpath(name).open("rb")


def _get_char_type(char_code: int) -> int:
    if 0 <= char_code < _MAX_CHAR_CODE:
        return _CHAR_TYPE_LOOKUP[char_code]
    return CHAR_TYPE_OTHER


def _get_abs_path(path: str) -> str:
    import os  # Import os here to ensure it's available

    return (
        os.path.normpath(path)
        if os.path.isabs(path)
        else os.path.normpath(os.path.join(os.getcwd(), path))
    )
