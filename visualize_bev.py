#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_bev.py
================
Inference & Visualization script for MetaDrive goal-prediction dataset samples.

Takes as input:
  - Path to a JSON file (.json), or dataset shard (.jsonl / .jsonl.gz) with --sample-idx

Visualizes in Bird's Eye View (BEV):
  1. MetaDrive road network & lane geometry (drivable areas, lane boundaries, centerlines)
  2. Ego vehicle (position, heading, velocity)
  3. Past trajectory history
  4. Reference lane path & checkpoints ahead
  5. Surrounding vehicles (positions & relative velocity vectors)
  6. Goal target position (future position at horizon T=5.0s)
  7. High-level instruction & dataset metadata HUD

Usage:
  python visualize_bev.py --json goal_dataset/smoke/train/shard_00000.jsonl --sample-idx 0 --out bev_visualization.png
  python visualize_bev.py --json sample.json --show
"""

import argparse
import gzip
import json
import math
import os
import sys
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import cv2

# Add local metadrive directory to sys.path if present
local_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadrive")
if os.path.exists(local_md) and local_md not in sys.path:
    sys.path.insert(0, local_md)

try:
    import pygame
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.constants import MetaDriveType, TopDownSemanticColor, PGDrivableAreaProperty
    from metadrive.obs.top_down_obs_impl import WorldSurface
    METADRIVE_AVAILABLE = True
except Exception as e:
    METADRIVE_AVAILABLE = False
    print(f"[Warning] MetaDrive import error: {e}")


# --------------------------------------------------------------------------------------
# Helper Functions & Transformations
# --------------------------------------------------------------------------------------
def ego_to_world(rel_pos: List[float], ego_pos: List[float], ego_heading: float) -> np.ndarray:
    """
    Transforms a vector from ego frame (x=forward, y=left) to MetaDrive world frame (x=right, y=up).
    """
    rx, ry = rel_pos[0], rel_pos[1]
    c, s = math.cos(ego_heading), math.sin(ego_heading)
    wx = ego_pos[0] + rx * c - ry * s
    wy = ego_pos[1] + rx * s + ry * c
    return np.array([wx, wy], dtype=float)


def load_sample(file_path: str, sample_idx: int = 0) -> Dict[str, Any]:
    """Loads a JSON sample record from .json, .jsonl, or .jsonl.gz file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".jsonl.gz"):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == sample_idx:
                    return json.loads(line)
            raise IndexError(f"Sample index {sample_idx} out of range in {file_path}")
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == sample_idx:
                    return json.loads(line)
            raise IndexError(f"Sample index {sample_idx} out of range in {file_path}")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if "\n" in content and content.startswith("{"):
                lines = content.splitlines()
                if sample_idx < len(lines):
                    return json.loads(lines[sample_idx])
            return json.loads(content)


# --------------------------------------------------------------------------------------
# MetaDrive BEV Renderer
# --------------------------------------------------------------------------------------
def render_metadrive_bev(
    sample: Dict[str, Any],
    film_size_px: int = 1200,
    default_crop_m: float = 90.0,
    predicted_goal_ego: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Renders a focused top-down view with lanes, vehicles, and both goal points.
    """
    meta = sample.get("meta", {})
    map_spec = meta.get("map", "S")
    if isinstance(map_spec, str) and map_spec.isdigit():
        map_spec = int(map_spec)
    scenario_id = sample.get("scenario_id", "seed_0")
    
    # Parse seed from scenario_id if present (e.g. "seed_1" -> 1)
    seed = 0
    if isinstance(scenario_id, str) and "seed_" in scenario_id:
        try:
            seed = int(scenario_id.split("seed_")[-1])
        except ValueError:
            seed = 0

    env_config = dict(
        use_render=False,
        start_seed=seed,
        num_scenarios=1,
        map=map_spec,
        traffic_density=meta.get("traffic_density", 0.1),
        traffic_mode=meta.get("traffic_mode", "trigger"),
        log_level=50,
    )

    env = MetaDriveEnv(env_config)
    try:
        env.reset(seed=seed)
        map_obj = env.current_map

        ego_state = sample["ego_state"]
        ego_pos = np.array(ego_state["position"], dtype=float)
        ego_heading = float(ego_state["heading"])

        # Determine target and predicted goal positions in world coordinates.
        label = sample.get("label", {})
        goal_wpos = label.get("future_position")
        if goal_wpos is None and "future_position_ego" in label:
            goal_wpos = ego_to_world(label["future_position_ego"], ego_pos, ego_heading)
        elif goal_wpos is not None:
            goal_wpos = np.array(goal_wpos, dtype=float)
        predicted_wpos = (
            ego_to_world(predicted_goal_ego, ego_pos, ego_heading)
            if predicted_goal_ego is not None
            else None
        )

        goal_positions = [
            position for position in (goal_wpos, predicted_wpos)
            if position is not None
        ]
        if goal_positions:
            center_pos = np.mean([ego_pos] + goal_positions, axis=0)
            dist_ego_goal = max(
                float(np.linalg.norm(position - ego_pos))
                for position in goal_positions
            )
        else:
            center_pos = ego_pos
            dist_ego_goal = 30.0

        view_extent_m = max(default_crop_m, dist_ego_goal * 2.2, 50.0)
        scaling = film_size_px / view_extent_m

        # Initialize Pygame WorldSurface
        surface = WorldSurface((film_size_px, film_size_px), 0, pygame.Surface((film_size_px, film_size_px)))
        surface.scaling = scaling
        surface.move_display_window_to(center_pos)

        # --------------------------------------------------
        # 1. Render MetaDrive Road Network
        # --------------------------------------------------
        all_lanes = map_obj.get_map_features(2)
        for obj in all_lanes.values():
            obj_type = obj.get("type")
            if MetaDriveType.is_lane(obj_type):
                if "polygon" in obj and len(obj["polygon"]) > 0:
                    pts = [surface.pos2pix(p[0], p[1]) for p in obj["polygon"]]
                    pygame.draw.polygon(surface, TopDownSemanticColor.get_color(obj_type), pts)
            elif MetaDriveType.is_road_line(obj_type) or MetaDriveType.is_road_boundary_line(obj_type):
                if "polyline" in obj and len(obj["polyline"]) > 1:
                    color = TopDownSemanticColor.get_color(obj_type)
                    stroke = max(2, surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2)
                    poly = obj["polyline"]
                    for i in range(len(poly) - 1):
                        pygame.draw.line(surface, color, surface.vec2pix(poly[i]), surface.vec2pix(poly[i + 1]), stroke)

        # --------------------------------------------------
        # 2. Render surrounding vehicles.
        # --------------------------------------------------
        other_vehicles = sample.get("lidar", {}).get("other_vehicles", [])
        for ov in other_vehicles:
            rel_pos = ov.get("rel_position", [0.0, 0.0])
            if abs(rel_pos[0]) < 1e-3 and abs(rel_pos[1]) < 1e-3:
                continue

            ov_wpos = ego_to_world(rel_pos, ego_pos, ego_heading)
            half_l, half_w = 2.25, 1.0
            raw_corners = [
                [half_l, half_w], [half_l, -half_w],
                [-half_l, -half_w], [-half_l, half_w],
            ]
            ov_heading = ego_heading
            ov_corners = [ego_to_world(c, ov_wpos, ov_heading) for c in raw_corners]
            ov_pix_corners = [surface.pos2pix(c[0], c[1]) for c in ov_corners]

            pygame.draw.polygon(surface, (180, 80, 230), ov_pix_corners)
            pygame.draw.polygon(surface, (255, 255, 255), ov_pix_corners, 1)

        # --------------------------------------------------
        # 3. Render ego vehicle.
        # --------------------------------------------------
        half_l, half_w = 2.3, 1.05
        raw_corners = [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
        ego_corners = [ego_to_world(c, ego_pos, ego_heading) for c in raw_corners]
        ego_pix_corners = [surface.pos2pix(c[0], c[1]) for c in ego_corners]

        # Draw vehicle body (Red)
        pygame.draw.polygon(surface, (235, 40, 40), ego_pix_corners)
        pygame.draw.polygon(surface, (255, 255, 255), ego_pix_corners, 2)

        # --------------------------------------------------
        # 4. Render target and predicted goals.
        # --------------------------------------------------
        ego_pix = surface.pos2pix(ego_pos[0], ego_pos[1])
        if goal_wpos is not None:
            goal_pix = surface.pos2pix(goal_wpos[0], goal_wpos[1])
            pygame.draw.circle(surface, (255, 215, 0), goal_pix, 16, 3)
            pygame.draw.circle(surface, (255, 215, 0), goal_pix, 5)

        if predicted_goal_ego is not None:
            predicted_pix = surface.pos2pix(
                predicted_wpos[0], predicted_wpos[1]
            )
            pygame.draw.circle(surface, (40, 180, 255), predicted_pix, 14, 3)
            pygame.draw.circle(surface, (40, 180, 255), predicted_pix, 4)

        # Convert pygame surface to OpenCV BGR image
        return WorldSurface.to_cv2_image(surface)

    finally:
        env.close()


# --------------------------------------------------------------------------------------
# HUD Banner Overlay
# --------------------------------------------------------------------------------------
def add_hud(img: np.ndarray, sample: Dict[str, Any]) -> np.ndarray:
    """Adds a compact metadata and focused legend to the visualization."""
    h, w, _ = img.shape
    bar_h = 100
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:, :] = img
    canvas[:bar_h, :] = (20, 20, 20)

    instr = sample.get("instruction", "N/A")
    ego_state = sample.get("ego_state", {})
    speed_ms = ego_state.get("speed", 0.0)
    speed_kmh = speed_ms * 3.6
    scenario_id = sample.get("scenario_id", "N/A")
    timestamp = sample.get("timestamp", 0.0)
    regime = sample.get("meta", {}).get("regime", "N/A")

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title & Instruction Banner
    cv2.putText(canvas, f"INSTRUCTION: {instr.upper()}", (20, 35), font, 0.85, (0, 230, 255), 2)
    cv2.putText(canvas, f"Scenario: {scenario_id} | t={timestamp:.1f}s", (w - 380, 35), font, 0.6, (220, 220, 220), 1)

    # Sub-info line
    cv2.putText(canvas, f"Ego Speed: {speed_ms:.1f} m/s ({speed_kmh:.1f} km/h) | Map: {regime}", (20, 65), font, 0.55, (200, 200, 200), 1)

    # Color Legend at bottom of top banner
    ly = 90
    items = [
        ("Ego Vehicle", (40, 40, 235)),
        ("Target goal", (0, 215, 255)),
        ("Predicted goal", (255, 180, 40)),
        ("Other Vehicles", (230, 80, 180)),
    ]
    
    lx = 20
    for label_text, bgr_color in items:
        cv2.circle(canvas, (lx, ly - 4), 6, bgr_color, -1)
        cv2.putText(canvas, label_text, (lx + 12, ly), font, 0.45, (230, 230, 230), 1)
        lx += 180

    return canvas


# --------------------------------------------------------------------------------------
# CLI Main
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MetaDrive BEV Dataset Visualizer & Goal Prediction Inference")
    parser.add_argument("--json", required=True, help="Path to input .json, .jsonl, or .jsonl.gz dataset file")
    parser.add_argument("--sample-idx", type=int, default=100, help="Record index if reading a .jsonl file")
    parser.add_argument("--out", default="bev_visualization.png", help="Path to save BEV output image")
    parser.add_argument("--crop-size", type=float, default=90.0, help="Default BEV view extent in meters")
    parser.add_argument("--show", action="store_true", help="Display image in interactive desktop window")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[visualize_bev] Loading record from {args.json} (sample index {args.sample_idx})...")
    sample = load_sample(args.json, args.sample_idx)

    print("\n--- SAMPLE SUMMARY ---")
    print(f"  ID         : {sample.get('id')}")
    print(f"  Instruction: {sample.get('instruction')}")
    print(f"  Ego State  : Pos={sample['ego_state']['position']}, Speed={sample['ego_state']['speed']} m/s")
    print(f"  Goal Label : {sample['label'].get('future_position')} (Ego Rel: {sample['label'].get('future_position_ego')})")
    print(f"  Meta Regime: {sample.get('meta', {}).get('regime')}")
    print("----------------------\n")

    if not METADRIVE_AVAILABLE:
        raise RuntimeError("MetaDrive environment is required for top-down BEV rendering.")

    print("[visualize_bev] Rendering MetaDrive BEV map...")
    img_bgr = render_metadrive_bev(sample, default_crop_m=args.crop_size)

    # Add HUD banner
    final_img = add_hud(img_bgr, sample)

    # Save output
    cv2.imwrite(args.out, final_img)
    print(f"[visualize_bev] Saved BEV visualization to -> {os.path.abspath(args.out)}")

    if args.show:
        try:
            cv2.imshow("MetaDrive BEV Visualization", final_img)
            print("[visualize_bev] Press any key in window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[visualize_bev] Window display not available (headless environment): {e}")


if __name__ == "__main__":
    main()
