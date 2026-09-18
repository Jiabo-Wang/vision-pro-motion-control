#!/usr/bin/env python3
"""按**文件路径**加载 vel_ar5 的 AR5Kinematics。

为什么不能直接 `from kinematics import AR5Kinematics`：

    >>> import kinematics
    >>> kinematics.__file__
    '.../site-packages/kinematics/__init__.py'

这个环境里装了一个**同名的 PyPI 包** `kinematics`。用模块名导入时，谁在
`sys.path` 上排前面谁赢 —— 于是能不能拿到正确的类，取决于调用方有没有记得
把 vel_ar5 的目录插到 `sys.path[0]`。忘了（或者被别的 import 重排了）就炸，
而且报错还分两种，看着像两个不同的毛病：

    ModuleNotFoundError: No module named 'kinematics'          # 路径没加
    ImportError: cannot import name 'AR5Kinematics' from ...   # 加了但被同名包顶掉

按绝对文件路径加载就没有这个问题：模块名叫什么、site-packages 里有什么，
都影响不到。vel_ar5 的 kinematics.py 只依赖 stdlib + numpy（没有相对导入），
所以可以这样独立加载。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

AR5_ROOT = Path("/home/crp-5070ti-01/yuhang_workspace/vel_ar5/openvla-energy/"
                "experiments/robot/ar5")

_CACHE = {}


def load_vel_module(name: str):
    """从 vel_ar5 的 ar5 目录按文件路径加载一个模块（只适用于无相对导入的）。"""
    if name in _CACHE:
        return _CACHE[name]
    path = AR5_ROOT / f"{name}.py"
    if not path.is_file():
        raise ImportError(
            f"找不到 {path}。vel_ar5 不在预期位置——如果仓库挪过，"
            f"改 {__file__} 里的 AR5_ROOT。")
    # 起一个**不会和任何已安装包撞名**的模块名
    spec = importlib.util.spec_from_file_location(f"_vel_ar5_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _CACHE[name] = mod
    return mod


def AR5Kinematics(*args, **kwargs):
    """vel_ar5 的 AR5Kinematics，按文件路径取。签名完全透传。"""
    return load_vel_module("kinematics").AR5Kinematics(*args, **kwargs)


def matrix_to_rpy(R):
    """vel_ar5 的 matrix_to_rpy（ROKAE 的 Rz·Ry·Rx 约定）。按文件路径取。

    和 AR5Kinematics 同理：顶层 `from transforms import ...` 依赖调用方先把
    vel_ar5 插进 sys.path，放在 import 段就必然比 sys.path.insert 早执行，
    直接 ModuleNotFoundError。transforms.py 只依赖 typing+numpy，可独立加载。
    """
    return load_vel_module("transforms").matrix_to_rpy(R)


def ar5_kinematics_class():
    """要类本身（而不是实例）时用这个，例如 isinstance 判断。"""
    return load_vel_module("kinematics").AR5Kinematics
