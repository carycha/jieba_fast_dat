import logging
import os
import sys
from collections.abc import Iterator
from typing import Any, BinaryIO

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)


def setLogLevel(log_level: int) -> None:
    default_logger.setLevel(log_level)


# Simplified for Python 3 only
text_type = str
string_types = (str,)
xrange = range


def iterkeys(d: dict[Any, Any]) -> Iterator[Any]:
    return iter(d.keys())


def itervalues(d: dict[Any, Any]) -> Iterator[Any]:
    return iter(d.values())


def iterms(d: dict[Any, Any]) -> Iterator[tuple[Any, Any]]:
    return iter(d.items())


def strdecode(sentence: str | bytes) -> str:
    # In Python 3, strings are unicode by default.
    # This function ensures the input is a string (unicode)
    # and handles potential byte string decoding if necessary.
    if isinstance(sentence, bytes):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


def get_module_res(*res: str) -> BinaryIO:
    # Assuming resources are directly accessible or handled by the C++ extension
    # For now, we'll return a file-like object for testing purposes if it's a path
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), *res))
    if os.path.exists(path):
        return open(path, "rb")
    else:
        # Fallback for resources that might be in the main jieba_fast_dat package root
        # This might need to be more robust depending on actual resource location
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path_in_root = os.path.normpath(os.path.join(root_path, *res))
        if os.path.exists(path_in_root):
            return open(path_in_root, "rb")
        else:
            raise FileNotFoundError(f"Resource not found: {path} or {path_in_root}")
