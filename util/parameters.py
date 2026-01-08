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
from collections.abc import MutableMapping
from typing import Any, Union, TextIO, Iterator
import math
import warnings

class JsonParameters(MutableMapping):
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

    def _as_dict(self) -> dict:
        if not isinstance(self.data, dict):
            raise TypeError(f"JsonParameters is not dict-like (data is {type(self.data).__name__})")
        return self.data

    def __getitem__(self, key: str) -> Any:
        return self._as_dict()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._as_dict()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._as_dict())

    def __len__(self) -> int:
        return len(self._as_dict())

class SpringDampingParameters(JsonParameters):
    """
    支持两套输入：
      A: mass + spring_coef(k) + damping_coef(c)
      B: mass + quality_factor(Q) + (omega_n 或 f_n)

    自动推导并补齐另一套参数；若冗余参数不一致则警告并退出。
    """

    # 允许的一些别名（可按你的 JSON 习惯继续扩展）
    _ALIASES = {
        "m": "mass",
        "k": "spring_coef",
        "b": "damping_coef",
        "Q": "quality_factor",
        "wn": "omega_n",
        "omega": "omega_n",
        "omega0": "omega_n",
        "w0": "omega_n",
        "fn": "f_n",
        "f0": "f_n",
    }

    def __init__(
        self,
        source: Union[str, os.PathLike, TextIO],
        *,
        encoding: str = "utf-8",
        rel_tol: float = 1e-6,
        abs_tol: float = 0.0,
        exit_on_conflict: bool = True,
    ):
        super().__init__(source, encoding=encoding)

        # 1) 形状约束：必须是 dict
        assert isinstance(self.data, dict), "SpringDampingParameters expects a JSON object (dict)."

        # 2) 规范化 key（别名 → 统一字段名）
        self._canonicalize_keys()

        # 3) 基本字段检查
        self._require("mass")
        self._require_positive("mass")

        # 4) 处理 omega_n / f_n 的互推与一致性校验
        self._reconcile_frequency(rel_tol=rel_tol, abs_tol=abs_tol, exit_on_conflict=exit_on_conflict)

        # 5) 判断输入组，并推导/校验
        has_kb = self._has_all("spring_coef", "damping_coef")
        has_Qw = self._has_all("quality_factor") and self._has_any("omega_n")  # reconcile 后，若 B 可用，一定有 omega_n

        if not has_kb and not has_Qw:
            self._conflict(
                "Parameters are incomplete. Provide either "
                "(spring_coef + damping_coef) or (quality_factor + omega_n/f_n), plus mass.",
                exit_on_conflict=exit_on_conflict,
            )

        if has_kb:
            self._require_positive("spring_coef")
            self._require_positive("damping_coef")

        if has_Qw:
            self._require_positive("quality_factor")
            self._require_positive("omega_n")

        # 冗余输入：两组都给了 → 校验一致性
        if has_kb and has_Qw:
            derived_from_kb = self._derive_from_kb()
            derived_from_Qw = self._derive_from_Qw()

            # 检查关键量一致性（你也可加更多字段）
            checks = [
                ("spring_coef", derived_from_Qw["spring_coef"], self.data["spring_coef"]),
                ("damping_coef", derived_from_Qw["damping_coef"], self.data["damping_coef"]),
                ("quality_factor", derived_from_kb["quality_factor"], self.data["quality_factor"]),
                ("omega_n", derived_from_kb["omega_n"], self.data["omega_n"]),
            ]
            bad = []
            for name, expected, given in checks:
                if not math.isclose(given, expected, rel_tol=rel_tol, abs_tol=abs_tol):
                    bad.append((name, given, expected))

            if bad:
                msg_lines = ["Conflicting parameters detected:"]
                for name, given, expected in bad:
                    msg_lines.append(f"  - {name}: given={given} expected≈{expected}")
                self._conflict("\n".join(msg_lines), exit_on_conflict=exit_on_conflict)

            # 一致：可选择提示冗余输入已通过校验
            warnings.warn("Spring-damping redundant parameters provided. Consistency check passed.", RuntimeWarning)

            # 用已有值为准即可；也可以统一覆盖为某一套的推导值（通常不必）
        elif has_kb and not has_Qw:
            # 用 A 推导 B 并补齐
            self.data.update(self._derive_from_kb())
        elif has_Qw and not has_kb:
            # 用 B 推导 A 并补齐
            self.data.update(self._derive_from_Qw())

        # 6) 额外派生量（可选）：阻尼比等
        # self.data.setdefault("damping_ratio", 1.0 / (2.0 * self.data["quality_factor"]))

    # ---------- helpers ----------

    def _canonicalize_keys(self):
        normalized = {}
        for k, v in self.data.items():
            kk = self._ALIASES.get(k, k)
            if kk in normalized:
                self._conflict(f"Duplicate keys after alias normalization: {k!r} maps to {kk!r}",
                               exit_on_conflict=True)
            normalized[kk] = v
        self.data = normalized

    def _require(self, key: str):
        if key not in self.data:
            self._conflict(f"Missing required key: {key}", exit_on_conflict=True)

    def _require_positive(self, key: str):
        self._require(key)
        val = self.data[key]
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
            self._conflict(f"{key} must be a positive finite number, got: {val!r}", exit_on_conflict=True)

    def _has_all(self, *keys: str) -> bool:
        return all(k in self.data for k in keys)

    def _has_any(self, *keys: str) -> bool:
        return any(k in self.data for k in keys)

    def _conflict(self, message: str, *, exit_on_conflict: bool):
        warnings.warn(message, RuntimeWarning)
        if exit_on_conflict:
            raise SystemExit(message)
        raise ValueError(message)

    def _reconcile_frequency(self, *, rel_tol: float, abs_tol: float, exit_on_conflict: bool):
        has_w = "omega_n" in self.data
        has_f = "f_n" in self.data

        if has_w:
            self._require_positive("omega_n")
        if has_f:
            self._require_positive("f_n")

        if has_w and has_f:
            expected_w = 2.0 * math.pi * float(self.data["f_n"])
            given_w = float(self.data["omega_n"])
            if not math.isclose(given_w, expected_w, rel_tol=rel_tol, abs_tol=abs_tol):
                self._conflict(
                    f"omega_n and f_n conflict: omega_n={given_w} but 2πf_n≈{expected_w}",
                    exit_on_conflict=exit_on_conflict,
                )
        elif (not has_w) and has_f:
            self.data["omega_n"] = 2.0 * math.pi * float(self.data["f_n"])
        elif has_w and (not has_f):
            self.data["f_n"] = float(self.data["omega_n"]) / (2.0 * math.pi)
        # 两者都没有：组 B 不可用（后面会判断）

    def _derive_from_kb(self) -> dict[str, float]:
        m = float(self.data["mass"])
        k = float(self.data["spring_coef"])
        c = float(self.data["damping_coef"])

        omega_n = math.sqrt(k / m)
        Q = (m * omega_n) / c  # Q = m*omega_n / c

        return {
            "omega_n": omega_n,
            "f_n": omega_n / (2.0 * math.pi),
            "quality_factor": Q,
        }

    def _derive_from_Qw(self) -> dict[str, float]:
        m = float(self.data["mass"])
        Q = float(self.data["quality_factor"])
        omega_n = float(self.data["omega_n"])

        k = m * omega_n * omega_n
        c = (m * omega_n) / Q

        return {
            "spring_coef": k,
            "damping_coef": c,
        }
