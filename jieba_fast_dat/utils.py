from __future__ import annotations

import importlib.resources
import os
import pickle
import re
import tempfile
from hashlib import md5
from pathlib import Path
from typing import IO, Any

__all__ = [
    "CHAR_TYPE_ZH",
    "CHAR_TYPE_NUM",
    "CHAR_TYPE_ALPHA",
    "CHAR_TYPE_OTHER",
    "RE_ENG",
    "RE_HAN_DEFAULT",
    "RE_SKIP_DEFAULT",
    "RE_ENG_NUM",
    "RE_HAN_DETAIL",
    "RE_SKIP_DETAIL",
    "RE_ENG_POS",
    "RE_NUM_POS",
    "RE_ENG_CHAR",
    "RE_HAN_FINALSEG",
    "RE_SKIP_FINALSEG",
    "get_module_res",
    "load_model_pickle",
    "strdecode",
    "CacheManager",
]

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


# Common Regex Patterns
RE_ENG = re.compile(r"[a-zA-Z0-9]", re.U)
RE_HAN_DEFAULT = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)
RE_SKIP_DEFAULT = re.compile(r"(\r\n|\s)", re.U)
RE_ENG_NUM = re.compile(r"^[a-zA-Z0-9]+(?:\.\d+)?%?$", re.U)

# POSSEG Regex Patterns
RE_HAN_DETAIL = re.compile(r"([\u4E00-\u9FD5]+)")
RE_SKIP_DETAIL = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
RE_ENG_POS = re.compile(r"[a-zA-Z0-9]+")
RE_NUM_POS = re.compile(r"[\.0-9]+")
RE_ENG_CHAR = re.compile("^[a-zA-Z0-9]$", re.U)

# FINALSEG Regex Patterns
RE_HAN_FINALSEG = re.compile("([\u4e00-\u9fd5]+)")
RE_SKIP_FINALSEG = re.compile("([a-zA-Z0-9]+(?:\\.\\d+)?%?)")


def get_module_res(module: str, name: str) -> IO[bytes]:
    """Get a binary resource file from a module."""
    return importlib.resources.files(module).joinpath(name).open("rb")


def load_model_pickle(module: str, filename: str) -> Any:
    """
    Common helper to load pickled model data from module resources.
    """
    with get_module_res(module, filename) as f:
        return pickle.loads(f.read())


def _get_char_type(char_code: int) -> int:
    """Internal helper to get character type ID."""
    if 0 <= char_code < _MAX_CHAR_CODE:
        return _CHAR_TYPE_LOOKUP[char_code]
    return CHAR_TYPE_OTHER


def _get_abs_path(path: str) -> str:
    """Get absolute path of a file string."""
    return (
        os.path.normpath(path)
        if os.path.isabs(path)
        else os.path.normpath(os.path.join(os.getcwd(), path))
    )


def strdecode(sentence: str | bytes) -> str:
    """
    Unified string decoding helper. Ensures the output is a string (unicode).
    """
    if isinstance(sentence, bytes):
        try:
            return sentence.decode("utf-8")
        except UnicodeDecodeError:
            return sentence.decode("gbk", "ignore")
    return str(sentence)


class CacheManager:
    """
    Unified manager for dictionary and model caches.
    """

    TMP_DIR = Path(tempfile.gettempdir())
    DEFAULT_PREFIX = "jieba_fast_dat"

    @classmethod
    def get_cache_path(
        cls,
        source_path: str | Path | None,
        prefix: str = DEFAULT_PREFIX,
        version: str = "v2",
    ) -> Path:
        """
        Generates a unique cache path based on the source path and version.
        """
        source_str = str(source_path) if source_path is not None else ""
        path_hash = md5(source_str.encode("utf-8")).hexdigest()
        return cls.TMP_DIR / f"{prefix}.{path_hash}.{version}.dat.cache"

    @classmethod
    def is_cache_valid(cls, source_path: str | Path | None, cache_path: Path) -> bool:
        """
        Checks if the cache is valid (exists and is newer than the source).
        """
        if not cache_path.exists():
            return False
        if source_path is None:  # Default dictionary case
            return True
        try:
            source_mtime = os.path.getmtime(str(source_path))
            return cache_path.stat().st_mtime > source_mtime
        except OSError:
            return False

    @classmethod
    def ensure_tmp_dir(cls) -> None:
        """Ensure the temporary directory exists."""
        cls.TMP_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_trie(
        cls,
        dat: Any,
        source_path: str | Path | None,
        prefix: str,
        cache_file: str | Path | None = None,
        logger: Any = None,
        force_rebuild: bool = False,
    ) -> bool:
        """
        Try to load trie from cache.
        """
        if force_rebuild:
            return False

        cache_path = (
            Path(cache_file)
            if cache_file
            else cls.get_cache_path(source_path, prefix=prefix)
        )

        if cls.is_cache_valid(source_path, cache_path):
            if logger:
                logger.debug(f"Loading model from cache {cache_path}")
            try:
                if dat.load_all(str(cache_path)) == 0:
                    return True
            except Exception as e:
                if logger:
                    logger.debug(f"Failed to load cache {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
        return False

    @classmethod
    def save_trie(
        cls,
        dat: Any,
        source_path: str | Path | None,
        prefix: str,
        cache_file: str | Path | None = None,
        logger: Any = None,
    ) -> None:
        """
        Save trie to cache.
        """
        cache_path = (
            Path(cache_file)
            if cache_file
            else cls.get_cache_path(source_path, prefix=prefix)
        )
        cls.ensure_tmp_dir()
        if logger:
            logger.debug(f"Dumping model to file cache {cache_path}")
        try:
            dat.save_all(str(cache_path))
            if logger:
                logger.debug("Dump cache file success.")
        except Exception as e:
            if logger:
                logger.exception(f"Dump cache file failed: {e}")
            cache_path.unlink(missing_ok=True)
