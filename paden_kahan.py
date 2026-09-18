#!/usr/bin/env python3
"""Paden-Kahan 子问题 —— 旋量法解析逆解的三块地基。

每个子问题都是一个**几何**问题，有闭式解和明确的解的个数（0/1/2），
这正是旋量法相对 DH 硬推的好处：分支从哪来、有几个，是看得见的。

约定：螺旋轴写成 (w, r)，w 是单位方向，r 是轴上任意一点。
      绕该轴转 θ 作用在点 p 上： p ↦ r + e^{[w]θ}(p − r)

  子问题 1: 绕**一个**轴把 p 转到 q            —— 0 或 1 解
  子问题 2: 绕**两个相交**轴把 p 转到 q        —— 0、1 或 2 解
  子问题 3: 绕一个轴把 p 转到「离 q 恰好 δ」   —— 0、1 或 2 解
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

EPS = 1e-9


def skew(w):
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rot(w, th):
    """Rodrigues：绕单位向量 w 转 th。"""
    W = skew(w)
    return np.eye(3) + np.sin(th) * W + (1.0 - np.cos(th)) * (W @ W)


def screw_apply(w, r, th, p):
    """绕过点 r、方向 w 的轴转 th，作用在点 p 上。"""
    r = np.asarray(r, dtype=float)
    return r + rot(w, th) @ (np.asarray(p, dtype=float) - r)


# ── 子问题 1 ────────────────────────────────────────────────────────────────
def subproblem1(w, r, p, q) -> List[float]:
    """绕单轴 (w,r) 把 p 转到 q。

    把 p、q 都投到垂直于 w 的平面上，剩下就是平面里的一个夹角。
    有解的条件：沿 w 的分量相等，且到轴的距离相等（转动不改变这两样）。
    """
    w = np.asarray(w, dtype=float)
    u = np.asarray(p, dtype=float) - np.asarray(r, dtype=float)
    v = np.asarray(q, dtype=float) - np.asarray(r, dtype=float)
    up = u - w * (w @ u)
    vp = v - w * (w @ v)
    if abs(w @ u - w @ v) > 1e-6 or abs(np.linalg.norm(up) - np.linalg.norm(vp)) > 1e-6:
        return []
    if np.linalg.norm(up) < EPS:
        return [0.0]                       # p 就在轴上，转多少都一样
    return [float(np.arctan2(w @ np.cross(up, vp), up @ vp))]


# ── 子问题 2 ────────────────────────────────────────────────────────────────
def subproblem2(w1, w2, r, p, q) -> List[Tuple[float, float]]:
    """两条**相交于 r** 的轴：先绕 w2 转 θ2，再绕 w1 转 θ1，把 p 送到 q。

        e^{[w1]θ1} e^{[w2]θ2} (p−r) = (q−r)

    几何：中间点 c 同时满足「到 w2 轴的距离 = p 的」和「到 w1 轴的距离 = q 的」，
    落在两个圆锥的交线上 —— 一般交出 **2** 个点，于是 2 组解。
    """
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.asarray(p, dtype=float) - r
    v = np.asarray(q, dtype=float) - r

    w1w2 = float(w1 @ w2)
    den = w1w2 ** 2 - 1.0
    if abs(den) < EPS:                     # 两轴平行，退化
        return []
    alpha = (w1w2 * (w2 @ u) - (w1 @ v)) / den
    beta = (w1w2 * (w1 @ v) - (w2 @ u)) / den
    cross = np.cross(w1, w2)
    n2 = float(cross @ cross)
    gsq = (float(u @ u) - alpha ** 2 - beta ** 2 - 2 * alpha * beta * w1w2) / n2
    if gsq < -1e-9:
        return []                          # 圆锥不相交 = 够不到
    gamma = float(np.sqrt(max(gsq, 0.0)))

    out = []
    for g in ({gamma, -gamma} if gamma > EPS else {0.0}):
        c = alpha * w1 + beta * w2 + g * cross
        t2 = subproblem1(w2, np.zeros(3), u, c)
        t1 = subproblem1(w1, np.zeros(3), c, v)
        if t2 and t1:
            out.append((t1[0], t2[0]))
    return out


# ── 子问题 3 ────────────────────────────────────────────────────────────────
def subproblem3(w, r, p, q, delta) -> List[float]:
    """绕单轴 (w,r) 转 p，使它**离点 q 恰好 delta**。

    这是余弦定理：投影到垂直 w 的平面后，两边长已知、第三边已知，求夹角。
    ± 两个解就是「肘上 / 肘下」的来源。
    """
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.asarray(p, dtype=float) - r
    v = np.asarray(q, dtype=float) - r
    up = u - w * (w @ u)
    vp = v - w * (w @ v)
    dsq = delta ** 2 - (w @ (np.asarray(p, dtype=float) - np.asarray(q, dtype=float))) ** 2
    if dsq < -1e-12:
        return []
    nu, nv = np.linalg.norm(up), np.linalg.norm(vp)
    if nu < EPS or nv < EPS:
        return []
    th0 = float(np.arctan2(w @ np.cross(up, vp), up @ vp))
    c = (nu ** 2 + nv ** 2 - max(dsq, 0.0)) / (2 * nu * nv)
    if not -1.0 - 1e-9 <= c <= 1.0 + 1e-9:
        return []                          # 够不到
    dth = float(np.arccos(np.clip(c, -1.0, 1.0)))
    return [th0 - dth] if dth < EPS else [th0 - dth, th0 + dth]
