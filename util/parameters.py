"""Utility for loading JSON-backed parameter sets.

This module provides a small helper class, Parameters, which loads JSON
data from a filesystem path or a file-like object. The class keeps the
underlying file handle when the instance opened the file itself and
will close it on demand (or when used as a context manager).

示例 / Example
    p = Parameters("config.json")
    print(p.data)

    # 或者使用上下文管理器
    with Parameters(open("config.json")) as p:
        print(p.data)

The class intentionally keeps a minimal API: .data holds the parsed
JSON structure, .close() will close the file if it was opened by the
instance, and __repr__/__str__ provide debug- and user-friendly
representations.
"""

import json
import io
import os
from typing import Any, Union, TextIO

class Parameters:
    """Load JSON parameters from a path or file-like object.

    Args:
        source: A filesystem path (str or os.PathLike) or a file-like
            object that implements .read(). If a path is provided, the
            Parameters instance opens the file and will close it when
            close() is called (or when used as a context manager). If a
            file-like object is provided, the caller is responsible for
            closing it.

    Attributes:
        data: The parsed JSON content (typically dict or list).
    """

    def __init__(self, source: Union[str, os.PathLike, TextIO], *, encoding: str = "utf-8"):
        # Source may be a path or a file-like object
        if isinstance(source, (str, os.PathLike)):
            self._fh = open(os.fspath(source), "r", encoding=encoding)
            self._opened_here = True
            self._path = os.fspath(source)
        elif hasattr(source, "read"):
            self._fh = source
            self._opened_here = False
            self._path = getattr(source, "name", None)
        else:
            raise TypeError("source must be a path or a file-like object with .read()")

        self.data = self.load_json()

    def load_json(self):
        """Read and parse JSON from the current file handle.

        If the file handle supports seek(), the pointer is moved to the
        start before reading to make behavior predictable for callers
        that supply an already-open file.
        """
        try:
            # Some file-like objects are not seekable; ignore errors.
            self._fh.seek(0)
        except Exception:
            pass
        return json.load(self._fh)

    def close(self):
        if self._opened_here:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # Do not suppress exceptions
        return False

    def __repr__(self):
        # 详细的、可复现的表示（调试用）
        path = self._path or "<file-like>"
        # 显示已加载数据的简短摘要（若为 dict 则显示键列表的前几个）
        summary = None
        if isinstance(self.data, dict):
            keys = list(self.data.keys())
            summary = f"keys={keys[:5]}{'...' if len(keys) > 5 else ''}"
        else:
            summary = type(self.data).__name__
        return f"Parameters(path={path!r}, opened_here={self._opened_here}, data={summary})"

    def __str__(self):
        # 更友好的字符串（print 调用时使用）
        path = self._path or "<file-like>"
        return f"Parameters from {path} — opened_here={self._opened_here}\n{json.dumps(self.data, indent=2)}"