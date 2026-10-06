#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_goal_dataset.py
=======================
Collects a goal-prediction dataset in MetaDrive (procedural-generation env, `MetaDriveEnv`).

For every sampled timestep it writes one JSON record (JSONL) containing:
  instruction | ego_state | lane_features | lidar (+ other_vehicles) | history | label (future position, 5 s)

CONVENTIONS (stick to them downstream)
  * units        : metres, seconds, m/s, radians
  * world frame  : MetaDrive world frame (x right, y up), heading = atan2(vy, vx)
  * ego frame    : x = forward, y = LEFT   (rel_position, rel_velocity, future_position_ego, checkpoints)
  * lateral      : positive = LEFT of lane centre. curvature positive = turning LEFT
  * steering     : normalised [-1, 1], positive = LEFT. `last_action` = [steer, throttle(+)/brake(-)] that
                   was applied on the previous step (logged in the left-positive convention, independent
                   of the sim's sign convention which is auto-calibrated at start-up).
  * dt           : decision_repeat * physics_step (0.1 s by default) -> timestamp = timestep * 0.1

INSTRUCTION LABEL (fixed rule, computed from what the ego actually does in the next `horizon` seconds)
  1. lane change : the ego's lane index changes on the same road segment during the horizon and the
                   accumulated heading change is small  -> lane_change_left / lane_change_right
                   (direction = side the new lane lies on)
  2. turn        : accumulated heading change over the horizon >= turn_thr (default 0.30 rad ~ 17 deg)
                   -> turn_left (heading increases) / turn_right (heading decreases)
  3. straight    : otherwise. compare mean speed over the last second of the horizon with current speed:
                   increase  > max(dv_abs, dv_rel*v) -> go_straight_fast
                   decrease  > same threshold        -> go_straight_slow
                   else                              -> go_straight

DIVERSITY
  * several map archetypes (multi-lane highway, curvy, X-/T-intersections, roundabouts, mixed, random)
  * traffic from empty to jammed, 'trigger' / 'respawn' traffic modes, random lane width / lane count
  * randomised driver styles (normal / aggressive / timid / weaver / stop-go) for the scripted expert:
      IDM car following, curvature-based speed limiting, random speed-target changes,
      gap-checked random + overtaking lane changes
  * class-balanced writing: per-class quota, per-episode per-class cap, adaptive boost of the lane-change
    and speed-change rates when those classes lag behind
  * train/val split by scenario seed (no leakage between splits)

USAGE
  pip install metadrive-simulator
  python collect_goal_dataset.py --smoke-test                       # verify your MetaDrive version first
  python collect_goal_dataset.py --out goal_dataset --samples-per-class 3000 --gzip
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.base_vehicle import BaseVehicle

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
INSTRUCTIONS = [
    "go_straight", "go_straight_fast", "go_straight_slow",
    "turn_left", "turn_right",
    "lane_change_left", "lane_change_right",
]
INSTRUCTION_ID = {n: i for i, n in enumerate(INSTRUCTIONS)}

PATH_STEP = 2.0          # spacing of sampled reference-path points (m)
WHEELBASE = 2.6          # for pure pursuit (m)
ACC_REF = 3.5            # accel (m/s^2) ~ throttle 1.0
BRK_REF = 7.0            # decel (m/s^2) ~ brake 1.0
JUNCTION_BLOCK_IDS = {"X", "T", "O"}

TRAFFIC_LEVELS = [("empty", 0.0), ("sparse", 0.06), ("normal", 0.12), ("dense", 0.22), ("jam", 0.35)]
MAP_ARCHETYPES = ["highway", "curvy", "urban_x", "urban_t", "roundabout", "mixed", "random_int"]


# --------------------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------------------
def wrap(a):
    """wrap angle(s) to [-pi, pi). works for floats and numpy arrays."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_ego(vec, heading):
    """world-frame vector -> ego frame (x forward, y left)."""
    c, s = math.cos(heading), math.sin(heading)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]], dtype=float)


def rnd(x, n=4):
    return round(float(x), n)


def rl(a, n=4):
    return [round(float(x), n) for x in a]


# --------------------------------------------------------------------------------------
# kinematics access (handles km/h vs m/s differences between MetaDrive versions)
# --------------------------------------------------------------------------------------
class Kin:
    def __init__(self, vel_scale: float = 1.0):
        self.vel_scale = vel_scale

    @staticmethod
    def pos(o):
        return np.asarray(o.position, dtype=float)[:2]

    def vel(self, o):
        return np.asarray(o.velocity, dtype=float)[:2] * self.vel_scale

    @staticmethod
    def heading(o):
        return float(o.heading_theta)


@dataclass
class Calib:
    steer_sign: float = 1.0   # multiply (left-positive) steering by this before sending to env
    vel_scale: float = 1.0    # multiply obj.velocity by this to obtain m/s


def calibrate(env, seed: int, dt: float) -> Calib:
    """Short probe: figure out (a) sign of steering, (b) whether obj.velocity is km/h or m/s."""
    out = env.reset(seed=seed)
    ego = env.agent
    h0 = float(ego.heading_theta)
    p_prev = np.asarray(ego.position, dtype=float)[:2]
    ratios = []
    for _ in range(14):
        res = env.step([0.3, 1.0])
        p = np.asarray(ego.position, dtype=float)[:2]
        fd = float(np.linalg.norm(p - p_prev)) / dt
        raw = float(np.linalg.norm(np.asarray(ego.velocity, dtype=float)[:2]))
        if fd > 1.0:
            ratios.append(raw / fd)
        p_prev = p
        done = (res[2] or res[3]) if len(res) == 5 else res[2]
        if done:
            break
    dh = wrap(float(ego.heading_theta) - h0)
    cal = Calib()
    if abs(dh) < 0.01:
        print("[calibrate] WARNING: heading barely changed, keeping steer_sign=+1")
    else:
        cal.steer_sign = 1.0 if dh > 0 else -1.0
    if ratios:
        r = float(np.median(ratios))
        cal.vel_scale = 1.0 / 3.6 if r > 2.0 else 1.0
    else:
        print("[calibrate] WARNING: no speed measurement, assuming velocity is in m/s")
    print(f"[calibrate] steer_sign={cal.steer_sign:+.0f}  vel_scale={cal.vel_scale:.3f}  dt={dt:.3f}s")
    return cal


# --------------------------------------------------------------------------------------
# road / lane geometry tools
# --------------------------------------------------------------------------------------
class RoadTools:
    def __init__(self, env):
        self.env = env
        self.graph: Dict[str, Dict[str, list]] = {}
        self.lane_index_of: Dict[int, Tuple[str, str, int]] = {}
        self.junction_edges = set()
        self.route: List[str] = []
        self.route_pos: Dict[str, int] = {}

    def rebuild(self, ego):
        """call after every env.reset(): the map changes with the seed."""
        cmap = self.env.engine.current_map
        self.graph = cmap.road_network.graph
        self.lane_index_of = {}
        for f, d in self.graph.items():
            for t, lanes in d.items():
                for i, ln in enumerate(lanes):
                    self.lane_index_of[id(ln)] = (f, t, i)
        self.junction_edges = set()
        try:
            for blk in cmap.blocks:
                if getattr(blk, "ID", None) in JUNCTION_BLOCK_IDS:
                    for f, d in blk.block_network.graph.items():
                        for t in d:
                            self.junction_edges.add((f, t))
        except Exception:
            pass  # is_intersection falls back to False
        try:
            self.route = list(ego.navigation.checkpoints)
        except Exception:
            self.route = []
        self.route_pos = {n: i for i, n in enumerate(self.route)}

    # ---- lane identity -------------------------------------------------------------
    def index(self, lane):
        return self.lane_index_of.get(id(lane))

    def is_junction(self, lane) -> bool:
        idx = self.index(lane)
        return bool(idx and (idx[0], idx[1]) in self.junction_edges)

    # ---- geometry ------------------------------------------------------------------
    @staticmethod
    def lat_long(lane, pos) -> Tuple[float, float, float]:
        """(longitudinal s, signed lateral (left +), lane heading at s). Convention-independent."""
        s, _ = lane.local_coordinates(pos)
        sc = min(max(float(s), 0.0), float(lane.length))
        p0 = np.asarray(lane.position(sc, 0.0), dtype=float)[:2]
        h = float(lane.heading_theta_at(sc))
        n = np.array([-math.sin(h), math.cos(h)])
        lat = float(np.dot(np.asarray(pos, dtype=float)[:2] - p0, n))
        return float(s), lat, h

    @staticmethod
    def curvature_at(lane, s, ds=4.0) -> float:
        L = float(lane.length)
        s0 = min(max(s - ds / 2, 0.0), max(L - ds, 0.0))
        s1 = min(s0 + ds, L)
        if s1 - s0 < 1e-3:
            return 0.0
        return float(wrap(lane.heading_theta_at(s1) - lane.heading_theta_at(s0)) / (s1 - s0))

    def lane_neighbors(self, lane, s):
        """neighbouring lanes of the same road segment, split into left / right, nearest first.
        each item: (|lateral distance|, lane_idx, lane_obj)"""
        idx = self.index(lane)
        if idx is None:
            return [], []
        f, t, i = idx
        left, right = [], []
        for j, other in enumerate(self.graph[f][t]):
            if j == i:
                continue
            p = np.asarray(other.position(min(max(s, 0.0), float(other.length)), 0.0), dtype=float)[:2]
            _, lat, _ = self.lat_long(lane, p)
            (left if lat > 0 else right).append((abs(lat), j, other))
        left.sort(key=lambda x: x[0])
        right.sort(key=lambda x: x[0])
        return left, right

    def successor(self, lane):
        """next lane along the planned route (geometrically nearest start point among candidates)."""
        idx = self.index(lane)
        if idx is None:
            return None
        to = idx[1]
        nxt_edges = self.graph.get(to, {})
        if not nxt_edges:
            return None
        cands = None
        i = self.route_pos.get(to)
        if i is not None and i + 1 < len(self.route):
            cands = nxt_edges.get(self.route[i + 1])
        if not cands:
            cands = [ln for lanes in nxt_edges.values() for ln in lanes]
        end = np.asarray(lane.position(float(lane.length), 0.0), dtype=float)[:2]
        return min(cands, key=lambda c: float(np.linalg.norm(
            np.asarray(c.position(0.0, 0.0), dtype=float)[:2] - end)))

    def build_path(self, lane, s0, length, step=PATH_STEP, max_hops=6):
        """centre-line polyline starting at (lane, s0), following the route across lane boundaries.
        returns (points[N,2], n_valid). trailing points are padded with the last real point."""
        n = int(length // step) + 1
        pts: List[np.ndarray] = []
        cur, s, hops = lane, max(float(s0), 0.0), 0
        while len(pts) < n:
            if s > cur.length:
                nxt = self.successor(cur)
                if nxt is None or hops >= max_hops:
                    break
                s -= float(cur.length)
                cur, hops = nxt, hops + 1
                continue
            pts.append(np.asarray(cur.position(s, 0.0), dtype=float)[:2])
            s += step
        if not pts:
            pts.append(np.asarray(lane.position(min(max(float(s0), 0.0), float(lane.length)), 0.0),
                                  dtype=float)[:2])
        nv = len(pts)
        while len(pts) < n:
            pts.append(pts[-1])
        return np.stack(pts), nv


def lane_width(lane, s) -> float:
    try:
        return float(lane.width_at(s))
    except Exception:
        return float(getattr(lane, "width", 3.5))


def path_curvature(P, nv, step=PATH_STEP):
    """signed curvature (left +) along a polyline, one value per point."""
    k = np.zeros(len(P))
    if nv >= 3:
        d = np.diff(P[:nv], axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        kk = wrap(np.diff(ang)) / step
        k[1:nv - 1] = kk
        k[0] = kk[0]
        k[nv - 1] = kk[-1]
    return k


def interp_path(path, nv, d):
    f = d / PATH_STEP
    i0 = int(min(math.floor(f), nv - 1))
    i1 = min(i0 + 1, nv - 1)
    a = 0.0 if i0 >= nv - 1 else min(max(f - i0, 0.0), 1.0)
    return path[i0] * (1.0 - a) + path[i1] * a


# --------------------------------------------------------------------------------------
# sensing helpers
# --------------------------------------------------------------------------------------
def list_vehicles(env, ego):
    eng = env.engine
    try:
        objs = list(eng.get_objects(lambda o: isinstance(o, BaseVehicle)).values())
    except Exception:
        tm = eng.traffic_manager
        objs = list(getattr(tm, "traffic_vehicles", None) or getattr(tm, "vehicles", []))
    return [o for o in objs if o is not ego]


def sense_others(env, ego, kin: Kin, max_range: float):
    p0, h0, v0 = kin.pos(ego), kin.heading(ego), kin.vel(ego)
    out = []
    for o in list_vehicles(env, ego):
        try:
            p, v = kin.pos(o), kin.vel(o)
        except Exception:
            continue
        d = float(np.linalg.norm(p - p0))
        if d > max_range:
            continue
        out.append(dict(pos=p, vel=v, dist=d, rel=to_ego(p - p0, h0), rel_vel=to_ego(v - v0, h0)))
    out.sort(key=lambda x: x["dist"])
    return out


def extract_lidar(obs, n: int) -> np.ndarray:
    """Obs layout with side-detector, lane-line detector and lidar `num_others` disabled is
    [ego+navigation state ..., lidar cloud points (n)] -> lidar = last n entries."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < n:
        return np.ones(n, dtype=np.float32)
    return np.clip(arr[-n:], 0.0, 1.0)


# --------------------------------------------------------------------------------------
# scripted expert driver
# --------------------------------------------------------------------------------------
@dataclass
class Style:
    name: str
    v_cruise: float
    a_max: float
    b_comf: float
    headway: float
    min_gap: float
    event_mean: float     # mean seconds between speed-target changes
    lc_rate: float        # lane-change attempts / second
    look_k: float
    max_thr: float
    lat_acc: float
    noise: float
    v_min: float = 3.0
    v_max: float = 22.0


def sample_style(rng: random.Random, ev_boost: float = 1.0) -> Style:
    u = rng.uniform
    mode = rng.choices(["normal", "aggressive", "timid", "weaver", "stop_go"],
                       weights=[3, 1.5, 1.5, 2.5, 1.5])[0]
    s = dict(name=mode, v_cruise=u(6, 18), a_max=u(1.5, 3.0), b_comf=u(2.0, 3.5), headway=u(1.1, 2.0),
             min_gap=u(2.0, 4.0), event_mean=u(5, 10), lc_rate=u(0.0, 0.03), look_k=u(0.5, 0.8),
             max_thr=u(0.7, 1.0), lat_acc=u(2.0, 3.5), noise=u(0.0, 0.03))
    if mode == "aggressive":
        s.update(v_cruise=u(14, 22), a_max=u(2.5, 3.5), headway=u(0.8, 1.2), lc_rate=u(0.04, 0.10),
                 lat_acc=u(3.0, 4.5), max_thr=1.0)
    elif mode == "timid":
        s.update(v_cruise=u(4, 10), a_max=u(1.0, 1.8), headway=u(1.8, 2.6), lc_rate=u(0.0, 0.01),
                 lat_acc=u(1.5, 2.2))
    elif mode == "weaver":
        s.update(lc_rate=u(0.08, 0.20))
    elif mode == "stop_go":
        s.update(event_mean=u(3, 5), v_cruise=u(5, 14))
    s["event_mean"] = s["event_mean"] / max(ev_boost, 1e-3)
    return Style(**s)


class ExpertDriver:
    """pure-pursuit steering + IDM longitudinal control + randomised speed targets + lane changes.
    act() returns [steer (left +, normalised), throttle(+)/brake(-)]."""

    def __init__(self, style: Style, rng: random.Random, tools: RoadTools, kin: Kin, dt: float,
                 max_steer_rad: float, lc_boost: float = 1.0):
        self.style, self.rng, self.tools, self.kin, self.dt = style, rng, tools, kin, dt
        self.max_steer_rad = max_steer_rad
        self.lc_boost = lc_boost
        self.t = 0.0
        self.v_target = style.v_cruise
        self.next_event = rng.uniform(1.0, style.event_mean)
        self.lc: Optional[dict] = None
        self.lc_cool = rng.uniform(2.0, 6.0)
        self.prev_steer = 0.0
        self.prev_thr = 0.0

    # ---- helpers -------------------------------------------------------------------
    def _speed_event(self):
        st = self.style
        for _ in range(8):
            new = self.v_target + self.rng.choice((-1.0, 1.0)) * self.rng.uniform(3.0, 9.0)
            if st.v_min <= new <= st.v_max:
                self.v_target = new
                return
        self.v_target = self.rng.uniform(st.v_min, st.v_max)

    def _lead(self, path, nv, others):
        P = path[:nv]
        best = None
        for o in others:
            if o["dist"] > 85.0 or o["rel"][0] < -0.5:
                continue
            d = np.linalg.norm(P - o["pos"], axis=1)
            k = int(np.argmin(d))
            if d[k] > 2.3:
                continue
            gap = max(k * PATH_STEP - 4.6, 0.2)
            if best is None or gap < best[0]:
                best = (gap, float(np.linalg.norm(o["vel"])))
        return (None, 0.0) if best is None else best

    def _plan(self, ref, pos, others):
        s, lat, hd = self.tools.lat_long(ref, pos)
        path, nv = self.tools.build_path(ref, s, 80.0)
        curv = path_curvature(path, nv)
        n30 = max(2, int(30.0 / PATH_STEP))
        kappa = float(np.max(np.abs(curv[:n30])))
        gap, v_lead = self._lead(path, nv, others)
        return dict(s=s, lat=lat, hd=hd, path=path, nv=nv, kappa=kappa, gap=gap, v_lead=v_lead)

    def _idm(self, v, v0, gap, v_lead):
        st = self.style
        a_free = max(st.a_max * (1.0 - (v / max(v0, 0.5)) ** 4), -2.0 * st.b_comf)
        if gap is None:
            return a_free
        dv = v - v_lead
        s_star = st.min_gap + max(0.0, v * st.headway + v * dv / (2.0 * math.sqrt(st.a_max * st.b_comf)))
        a = a_free - st.a_max * (s_star / max(gap, 0.3)) ** 2
        return float(np.clip(a, -9.0, st.a_max))

    # ---- lane change ---------------------------------------------------------------
    def _lc_safe(self, tgt, pos, v, others) -> bool:
        s_e, _, _ = self.tools.lat_long(tgt, pos)
        for o in others:
            if o["dist"] > 70.0:
                continue
            so, lo, _ = self.tools.lat_long(tgt, o["pos"])
            if abs(lo) > 2.4:
                continue
            gap = so - s_e
            vo = float(np.linalg.norm(o["vel"]))
            front_need = 12.0 + max(0.0, v - vo) * 2.0
            rear_need = 9.0 + max(0.0, vo - v) * 2.5
            if -rear_need < gap < front_need:
                return False
        return True

    def _try_start_lc(self, lane, plan, v, pos, others):
        st = self.style
        if self.lc is not None or self.lc_cool > 0.0:
            return
        s = plan["s"]
        remaining = float(lane.length) - s
        if v < 2.0 or remaining < 15.0 or plan["kappa"] > 0.008 or self.tools.is_junction(lane):
            return
        nxt = self.tools.successor(lane)
        if nxt is not None and self.tools.is_junction(nxt) and remaining < 60.0:
            return
        left, right = self.tools.lane_neighbors(lane, s)
        opts = []
        if left:
            opts.append((+1, left[0][2]))
        if right:
            opts.append((-1, right[0][2]))
        if not opts:
            return
        blocked = plan["gap"] is not None and plan["gap"] < 35.0 and plan["v_lead"] < self.v_target - 2.0
        p = st.lc_rate * self.lc_boost * self.dt
        if not (self.rng.random() < p or (blocked and self.rng.random() < 0.05)):
            return
        self.rng.shuffle(opts)
        for d, tgt in opts:
            if self._lc_safe(tgt, pos, v, others):
                self.lc = dict(dir=d, lane=tgt, t0=self.t)
                return

    def _advance_lc(self, pos, heading):
        lc = self.lc
        tgt = lc["lane"]
        s, lat, hd = self.tools.lat_long(tgt, pos)
        if s > tgt.length - 1.0:
            nxt = self.tools.successor(tgt)
            if nxt is None:
                self.lc, self.lc_cool = None, self.rng.uniform(4.0, 8.0)
                return None
            lc["lane"] = tgt = nxt
            s, lat, hd = self.tools.lat_long(tgt, pos)
        done = (abs(lat) < 0.35 and abs(wrap(heading - hd)) < 0.10) or (self.t - lc["t0"] > 8.0)
        if done:
            self.lc, self.lc_cool = None, self.rng.uniform(4.0, 10.0)
            return None
        return tgt

    # ---- main ----------------------------------------------------------------------
    def act(self, ego, others) -> np.ndarray:
        st, dt = self.style, self.dt
        self.t += dt
        self.lc_cool -= dt
        pos, h = self.kin.pos(ego), self.kin.heading(ego)
        v = float(np.linalg.norm(self.kin.vel(ego)))
        lane = ego.lane

        if self.t >= self.next_event:
            self._speed_event()
            self.next_event = self.t + self.rng.expovariate(1.0 / st.event_mean)

        ref = lane
        if self.lc is not None:
            ref = self._advance_lc(pos, h) or lane
        plan = self._plan(ref, pos, others)
        if self.lc is None:
            self._try_start_lc(lane, plan, v, pos, others)
            if self.lc is not None:
                ref = self.lc["lane"]
                plan = self._plan(ref, pos, others)

        # ---- longitudinal (IDM) ----
        v0 = max(min(self.v_target, math.sqrt(st.lat_acc / max(plan["kappa"], 1e-3))), 3.5)
        a = self._idm(v, v0, plan["gap"], plan["v_lead"])
        cmd_thr = a / ACC_REF if a >= 0 else a / BRK_REF
        cmd_thr = float(np.clip(cmd_thr, -1.0, st.max_thr))
        thr = self.prev_thr + float(np.clip(cmd_thr - self.prev_thr, -0.5, 0.2))

        # ---- lateral (pure pursuit on reference path) ----
        Ld = float(np.clip(st.look_k * v + 4.0, 5.0, 22.0))
        if self.lc is not None:
            Ld = max(Ld, 8.0 + 0.6 * v)
        tgt = interp_path(plan["path"], plan["nv"], Ld)
        tx, ty = to_ego(tgt - pos, h)
        ld = max(math.hypot(tx, ty), 1.0)
        alpha = math.atan2(ty, tx)
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
        cmd_st = float(np.clip(delta / self.max_steer_rad, -1.0, 1.0))
        steer = self.prev_steer + float(np.clip(cmd_st - self.prev_steer, -0.25, 0.25))

        if st.noise > 0:
            steer += self.rng.gauss(0.0, st.noise)
            thr += self.rng.gauss(0.0, st.noise)
        steer = float(np.clip(steer, -1.0, 1.0))
        thr = float(np.clip(thr, -1.0, 1.0))
        self.prev_steer, self.prev_thr = steer, thr
        return np.array([steer, thr], dtype=float)


# --------------------------------------------------------------------------------------
# instruction labelling (fixed rules)
# --------------------------------------------------------------------------------------
def label_instruction(frames, k: int, j: int, cfg) -> str:
    mk = frames[k]["_m"]
    net, first, dpsi = 0, 0, 0.0
    for i in range(k, j):
        a, b = frames[i]["_m"], frames[i + 1]["_m"]
        dpsi += float(wrap(b["heading"] - a["heading"]))      # unwrapped accumulated heading change
        if a["edge"] is not None and a["edge"] == b["edge"] and a["lane_i"] != b["lane_i"]:
            d = int(np.sign(b["n_right"] - a["n_right"]))      # more lanes on the right -> moved LEFT
            net += d
            if first == 0:
                first = d
    lc = int(np.sign(net)) if net != 0 else first

    if lc != 0 and abs(dpsi) < cfg.turn_thr:
        return "lane_change_left" if lc > 0 else "lane_change_right"
    if abs(dpsi) >= cfg.turn_thr:
        return "turn_left" if dpsi > 0 else "turn_right"

    v0 = mk["speed"]
    v_end = float(np.mean([frames[i]["_m"]["speed"] for i in range(max(k, j - 9), j + 1)]))
    dv = v_end - v0
    thr = max(cfg.dv_abs, cfg.dv_rel * max(v0, 5.0))
    if dv > thr:
        return "go_straight_fast"
    if dv < -thr:
        return "go_straight_slow"
    return "go_straight"


# --------------------------------------------------------------------------------------
# episode runner: rollout -> frames -> labelled samples
# --------------------------------------------------------------------------------------
class EpisodeRunner:
    def __init__(self, env, cfg, calib: Calib, dt: float, rng: random.Random, regime: dict):
        self.env, self.cfg, self.calib, self.dt, self.rng, self.regime = env, cfg, calib, dt, rng, regime
        self.kin = Kin(calib.vel_scale)
        self.tools = RoadTools(env)

    # ---- one timestep snapshot -----------------------------------------------------
    def _snapshot(self, seed, step, obs, last_action, prev_h, others):
        cfg, tools, kin = self.cfg, self.tools, self.kin
        ego = self.env.agent
        pos, h, vel = kin.pos(ego), kin.heading(ego), kin.vel(ego)
        speed = float(np.linalg.norm(vel))
        yaw_rate = 0.0 if prev_h is None else float(wrap(h - prev_h) / self.dt)
        lane = ego.lane
        s, lat, lane_h = tools.lat_long(lane, pos)
        w = lane_width(lane, s)
        left, right = tools.lane_neighbors(lane, s)
        idx = tools.index(lane)
        edge = (idx[0], idx[1]) if idx else None
        lane_i = idx[2] if idx else -1

        lane_features = dict(
            dist_to_left_border=rnd(w / 2 - lat, 3),
            dist_to_right_border=rnd(w / 2 + lat, 3),
            lane_width=rnd(w, 3),
            curvature=rnd(tools.curvature_at(lane, s), 5),
            is_intersection=bool(tools.is_junction(lane)),
            num_lanes_left=len(left),
            num_lanes_right=len(right),
        )
        if cfg.checkpoints:
            path, nv = tools.build_path(lane, s, max(cfg.checkpoint_dists) + PATH_STEP)
            curv = path_curvature(path, nv)
            cps = []
            for d in cfg.checkpoint_dists:
                i = min(int(round(d / PATH_STEP)), len(path) - 1)
                rel = to_ego(path[i] - pos, h)
                cps.append(dict(rel_x=rnd(rel[0], 2), rel_y=rnd(rel[1], 2), curvature=rnd(curv[i], 4)))
            lane_features["checkpoints"] = cps

        near = [o for o in others if o["dist"] <= cfg.lidar_range][:cfg.num_others]
        ov = [dict(rel_position=rl(o["rel"], 2), rel_velocity=rl(o["rel_vel"], 2)) for o in near]
        while len(ov) < cfg.num_others:
            ov.append(dict(rel_position=[0.0, 0.0], rel_velocity=[0.0, 0.0]))

        cloud = np.round(extract_lidar(obs, cfg.num_lasers), 3).tolist()
        return dict(
            id=f"scenario_{seed:05d}_t_{step}",
            scenario_id=f"seed_{seed}",
            timestep=step,
            timestamp=rnd(step * self.dt, 3),
            ego_state=dict(
                speed=rnd(speed, 3),
                steering=rnd(last_action[0], 3),
                yaw_rate=rnd(yaw_rate, 4),
                heading=rnd(h, 4),
                position=rl(pos, 3),
                velocity=rl(vel, 3),
                last_action=rl(last_action, 3),
                lateral_offset=rnd(lat, 3),
                heading_diff=rnd(wrap(h - lane_h), 4),
            ),
            lane_features=lane_features,
            lidar=dict(cloud_points=cloud, num_lasers=cfg.num_lasers, max_range=cfg.lidar_range,
                       other_vehicles=ov),
            _m=dict(heading=h, speed=speed, pos=pos, edge=edge, lane_i=lane_i, n_right=len(right)),
        )

    # ---- rollout -------------------------------------------------------------------
    def rollout(self, seed: int, style: Style, lc_boost: float) -> List[dict]:
        env, cfg = self.env, self.cfg
        out = env.reset(seed=seed)
        obs = out[0] if isinstance(out, tuple) else out
        ego = env.agent
        self.tools.rebuild(ego)
        try:
            max_steer_rad = math.radians(float(ego.config["max_steering"]))
        except Exception:
            max_steer_rad = math.radians(40.0)
        driver = ExpertDriver(style, self.rng, self.tools, self.kin, self.dt, max_steer_rad, lc_boost)

        frames: List[dict] = []
        last_action, prev_h, slow = np.zeros(2), None, 0
        for step in range(cfg.max_steps):
            others = sense_others(env, ego, self.kin, max(cfg.lidar_range, 85.0))
            fr = self._snapshot(seed, step, obs, last_action, prev_h, others)
            frames.append(fr)
            prev_h = fr["_m"]["heading"]

            action = driver.act(ego, others)
            res = env.step([self.calib.steer_sign * float(action[0]), float(action[1])])
            if len(res) == 5:
                obs, _, term, trunc, _ = res
                done = bool(term or trunc)
            else:
                obs, _, done, _ = res
            last_action = action
            if done:
                break
            slow = slow + 1 if (step > 30 and fr["_m"]["speed"] < 0.3) else 0
            if slow > 150:       # stuck (e.g. blocked by a stopped car)
                break
        return frames

    # ---- labelling -----------------------------------------------------------------
    def finalize(self, frames: List[dict], seed: int, style: Style) -> List[dict]:
        cfg, dt = self.cfg, self.dt
        H = int(round(cfg.horizon_s / dt))
        n = len(frames)
        if n <= H + 1:
            return []
        phase = self.rng.randrange(cfg.stride)
        skip = int(cfg.skip_start_s / dt)
        cands: Dict[str, List[int]] = defaultdict(list)
        for k in range(skip, n - H):
            if (k % cfg.stride) != phase:
                continue
            j = k + H
            if max(frames[i]["_m"]["speed"] for i in range(k, j + 1, 5)) < 0.5:
                continue         # stationary window
            cands[label_instruction(frames, k, j, cfg)].append(k)

        chosen: List[Tuple[int, str]] = []
        for c, ks in cands.items():
            self.rng.shuffle(ks)
            chosen += [(k, c) for k in ks[:cfg.max_per_episode_class]]
        chosen.sort()
        meta = dict(regime=self.regime["name"], map=str(self.regime["map"]),
                    traffic_density=self.regime["density"], traffic_mode=self.regime["traffic_mode"],
                    driver_style=style.name, dt=dt)
        return [self._build_sample(frames, k, k + H, c, meta) for k, c in chosen]

    def _build_sample(self, frames, k, j, instr, meta) -> dict:
        cfg = self.cfg
        f, mk, mj = frames[k], frames[k]["_m"], frames[j]["_m"]
        hist_idx = [max(0, k - (cfg.history - 1) + i) for i in range(cfg.history)]
        history = dict(
            ego_positions=[frames[i]["ego_state"]["position"] for i in hist_idx],
            ego_speeds=[frames[i]["ego_state"]["speed"] for i in hist_idx],
        )
        if cfg.lidar_history:
            history["lidar_cloud_points"] = [frames[i]["lidar"]["cloud_points"] for i in hist_idx]
        rel = to_ego(mj["pos"] - mk["pos"], mk["heading"])
        label = dict(
            future_position=rl(mj["pos"], 3),
            future_position_ego=rl(rel, 3),
            future_heading=rnd(mj["heading"], 4),
            future_heading_ego=rnd(wrap(mj["heading"] - mk["heading"]), 4),
            horizon=cfg.horizon_s,
        )
        return dict(
            id=f["id"], scenario_id=f["scenario_id"], timestep=f["timestep"], timestamp=f["timestamp"],
            instruction=instr, instruction_id=INSTRUCTION_ID[instr],
            ego_state=f["ego_state"], lane_features=f["lane_features"], lidar=f["lidar"],
            history=history, label=label, meta=meta,
        )


# --------------------------------------------------------------------------------------
# dataset writer with per-class quotas
# --------------------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, root, split, shard_size, use_gz):
        self.dir = os.path.join(root, split)
        os.makedirs(self.dir, exist_ok=True)
        self.shard_size, self.use_gz = shard_size, use_gz
        self.n_in_shard, self.shard_idx, self.fh = 0, 0, None

    def _open(self):
        ext = ".jsonl.gz" if self.use_gz else ".jsonl"
        path = os.path.join(self.dir, f"shard_{self.shard_idx:05d}{ext}")
        self.fh = gzip.open(path, "wt", encoding="utf-8") if self.use_gz else open(path, "w", encoding="utf-8")
        self.n_in_shard = 0

    def write(self, sample):
        if self.fh is None or self.n_in_shard >= self.shard_size:
            self.close()
            self._open()
            self.shard_idx += 1
        self.fh.write(json.dumps(sample, separators=(",", ":")) + "\n")
        self.n_in_shard += 1

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.out, exist_ok=True)
        self.quota = cfg.samples_per_class
        self.counts = Counter()
        self.by_regime: Dict[str, Counter] = defaultdict(Counter)
        self.by_style: Dict[str, Counter] = defaultdict(Counter)
        self.split_counts = Counter()
        self.writers = {s: ShardWriter(cfg.out, s, cfg.shard_size, cfg.gzip) for s in ("train", "val")}
        self.last_sample = None

    def fill(self, c) -> float:
        return min(1.0, self.counts[c] / max(self.quota, 1))

    def full(self) -> bool:
        return all(self.counts[c] >= self.quota for c in INSTRUCTIONS)

    def split_of(self, seed) -> str:
        return "val" if (seed * 7919) % 100 < self.cfg.val_pct else "train"

    def add_episode(self, samples, seed) -> int:
        split, added = self.split_of(seed), 0
        for s in samples:
            c = s["instruction"]
            if self.counts[c] >= self.quota:
                continue
            self.writers[split].write(s)
            self.counts[c] += 1
            self.split_counts[split] += 1
            self.by_regime[s["meta"]["regime"]][c] += 1
            self.by_style[s["meta"]["driver_style"]][c] += 1
            self.last_sample = s
            added += 1
        return added

    def stats(self) -> dict:
        tot = sum(self.counts.values())
        return dict(
            total=tot, per_class={c: self.counts[c] for c in INSTRUCTIONS},
            per_class_fraction={c: round(self.counts[c] / max(tot, 1), 4) for c in INSTRUCTIONS},
            splits=dict(self.split_counts),
            per_regime={k: dict(v) for k, v in self.by_regime.items()},
            per_driver_style={k: dict(v) for k, v in self.by_style.items()},
        )

    def close(self):
        for w in self.writers.values():
            w.close()
        with open(os.path.join(self.cfg.out, "stats.json"), "w") as f:
            json.dump(self.stats(), f, indent=2)


# --------------------------------------------------------------------------------------
# regimes (map x traffic) and env construction
# --------------------------------------------------------------------------------------
def make_regime(rng: random.Random, i: int) -> dict:
    arch = MAP_ARCHETYPES[i % len(MAP_ARCHETYPES)]
    tname, dens = rng.choice(TRAFFIC_LEVELS)
    if arch == "highway":
        m = "".join(rng.choices("SC", weights=[4, 1], k=rng.randint(4, 6)))
    elif arch == "curvy":
        m = "".join(rng.choices("CS", weights=[3, 1], k=rng.randint(4, 6)))
    elif arch == "urban_x":
        m = "".join(rng.choice(["SX", "CX", "SXS"]) for _ in range(rng.randint(2, 3)))
    elif arch == "urban_t":
        m = "".join(rng.choice(["ST", "CT", "STS"]) for _ in range(rng.randint(2, 3)))
    elif arch == "roundabout":
        m = rng.choice(["SOS", "SOC", "COS"]) + rng.choice(["", "S", "C"])
    elif arch == "mixed":
        m = "".join(rng.choices("SCXTO", weights=[3, 3, 2, 1.5, 0.7], k=rng.randint(5, 7)))
    else:
        m = rng.randint(4, 7)
    return dict(name=f"{arch}|{tname}", map=m, density=dens,
                traffic_mode=rng.choice(["trigger", "respawn"]))


def make_env(cfg, regime, seed0, n_ep):
    config = dict(
        use_render=cfg.render,
        num_scenarios=n_ep,
        start_seed=seed0,
        map=regime["map"],
        traffic_density=regime["density"],
        traffic_mode=regime["traffic_mode"],
        random_lane_width=True,
        random_lane_num=True,
        accident_prob=cfg.accident_prob,
        horizon=cfg.max_steps,
        out_of_road_done=True,
        crash_vehicle_done=True,
        crash_object_done=True,
        on_continuous_line_done=False,
        log_level=logging.ERROR,
        vehicle_config=dict(
            lidar=dict(num_lasers=cfg.num_lasers, distance=cfg.lidar_range, num_others=0),
            side_detector=dict(num_lasers=0),
            lane_line_detector=dict(num_lasers=0),
            show_navi_mark=False,
        ),
    )
    return MetaDriveEnv(config)


def run_regime(cfg, regime, seed0, n_ep, rng, ds: Dataset, calib: Optional[Calib]):
    env = make_env(cfg, regime, seed0, n_ep)
    n_fail, n_done = 0, 0
    try:
        dt = float(env.config["decision_repeat"]) * float(env.config["physics_world_step_size"])
        if calib is None:
            calib = calibrate(env, seed0, dt)
        runner = EpisodeRunner(env, cfg, calib, dt, rng, regime)
        for ep in range(n_ep):
            seed = seed0 + ep
            if ds.full():
                break
            lc_fill = min(ds.fill("lane_change_left"), ds.fill("lane_change_right"))
            sp_fill = min(ds.fill("go_straight_fast"), ds.fill("go_straight_slow"))
            lc_boost = 1.0 + 2.0 * (1.0 - lc_fill)
            ev_boost = 1.0 + 1.0 * (1.0 - sp_fill)
            style = sample_style(rng, ev_boost)
            try:
                frames = runner.rollout(seed, style, lc_boost)
                samples = runner.finalize(frames, seed, style)
                added = ds.add_episode(samples, seed)
                n_done += 1
                print(f"  seed {seed:6d} | {regime['name']:<22} map={str(regime['map']):<10} "
                      f"style={style.name:<10} frames={len(frames):5d} cand={len(samples):4d} added={added:4d}")
            except Exception:
                n_fail += 1
                print(f"  seed {seed}: episode failed\n{traceback.format_exc()}")
                if n_fail >= 3:
                    print("  too many failures in this regime, skipping the rest")
                    break
    finally:
        env.close()
    return calib, n_done


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MetaDrive goal-prediction dataset collector")
    p.add_argument("--out", default="goal_dataset")
    p.add_argument("--samples-per-class", type=int, default=3000, help="quota for each of the 7 instructions")
    p.add_argument("--max-episodes", type=int, default=100000)
    p.add_argument("--episodes-per-env", type=int, default=10, help="episodes (seeds) per env/regime instance")
    p.add_argument("--max-steps", type=int, default=1500, help="env steps per episode (0.1 s each)")
    p.add_argument("--horizon-s", type=float, default=5.0)
    p.add_argument("--history", type=int, default=3, help="#frames of history incl. current")
    p.add_argument("--no-lidar-history", dest="lidar_history", action="store_false")
    p.add_argument("--num-lasers", type=int, default=240)
    p.add_argument("--lidar-range", type=float, default=50.0)
    p.add_argument("--num-others", type=int, default=6)
    p.add_argument("--no-checkpoints", dest="checkpoints", action="store_false",
                   help="drop route checkpoints from lane_features")
    p.add_argument("--checkpoint-dists", type=float, nargs="+", default=[15.0, 30.0, 50.0])
    p.add_argument("--stride", type=int, default=2, help="keep every n-th frame")
    p.add_argument("--skip-start-s", type=float, default=1.0)
    p.add_argument("--max-per-episode-class", type=int, default=150)
    p.add_argument("--turn-thr", type=float, default=0.30, help="rad of heading change over horizon")
    p.add_argument("--dv-abs", type=float, default=2.0, help="m/s speed change for fast/slow")
    p.add_argument("--dv-rel", type=float, default=0.15)
    p.add_argument("--accident-prob", type=float, default=0.0)
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--gzip", action="store_true")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="python RNG seed")
    p.add_argument("--render", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="2 short episodes, print one sample, exit")
    return p.parse_args()


def main():
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.smoke_test:
        cfg.out = os.path.join(cfg.out, "smoke")
        cfg.max_episodes, cfg.episodes_per_env, cfg.max_steps = 2, 2, 400
        cfg.samples_per_class = 10 ** 6
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    ds = Dataset(cfg)
    calib, seed_ctr, ep_total, round_i, t0 = None, cfg.start_seed, 0, 0, time.time()
    try:
        while not ds.full() and ep_total < cfg.max_episodes:
            regime = make_regime(rng, round_i)
            round_i += 1
            n_ep = min(cfg.episodes_per_env, cfg.max_episodes - ep_total)
            print(f"\n[regime {round_i}] {regime}  seeds {seed_ctr}..{seed_ctr + n_ep - 1}")
            try:
                calib, n_done = run_regime(cfg, regime, seed_ctr, n_ep, rng, ds, calib)
            except Exception:
                print("regime failed (invalid map string / config for this MetaDrive version?):")
                print(traceback.format_exc())
                n_done = 0
                if round_i > 3 * len(MAP_ARCHETYPES) and ep_total == 0:
                    print("nothing works, aborting")
                    break
            seed_ctr += n_ep
            ep_total += n_done
            el = time.time() - t0
            print(f"[progress] {ep_total} episodes | {el / 60:.1f} min | " +
                  " ".join(f"{c}:{ds.counts[c]}" for c in INSTRUCTIONS))
            if cfg.smoke_test:
                break
    except KeyboardInterrupt:
        print("interrupted - flushing what has been collected")
    finally:
        ds.close()

    st = ds.stats()
    print("\n===== FINAL CLASS DISTRIBUTION =====")
    for c in INSTRUCTIONS:
        print(f"  {c:<18} {st['per_class'][c]:7d}   {100 * st['per_class_fraction'][c]:5.1f}%")
    print(f"  total {st['total']}  splits {st['splits']}  -> {cfg.out}")

    if cfg.smoke_test and ds.last_sample is not None:
        s = json.loads(json.dumps(ds.last_sample))
        s["lidar"]["cloud_points"] = s["lidar"]["cloud_points"][:8] + ["..."]
        if "lidar_cloud_points" in s["history"]:
            s["history"]["lidar_cloud_points"] = "[... omitted ...]"
        print("\nSample record (lidar truncated):")
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()