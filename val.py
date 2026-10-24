import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from train import (
    GoalPredictionDataset,
    GoalPointPredictor,
    NormalizationStats,
    create_dataloader,
)

LOCAL_METADRIVE_DIR = Path(__file__).parent / "metadrive"
if LOCAL_METADRIVE_DIR.exists() and str(LOCAL_METADRIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_METADRIVE_DIR))


PROJECT_DIR = Path(__file__).parent
VALIDATION_DATASET_DIR = PROJECT_DIR / "goal_dataset" / "val"
CHECKPOINT_PATH = PROJECT_DIR / "model" / "goal_point_predictor.pt"


def load_checkpoint(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[GoalPointPredictor, NormalizationStats]:
    """Load the trained model and the training normalization statistics."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. Run train_model() first."
        )

    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location=selected_device,
    )
    model = GoalPointPredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    stats = NormalizationStats(**checkpoint["normalization_stats"])
    return model, stats


@torch.no_grad()
def evaluate(
    validation_dataset_dir: str | Path = VALIDATION_DATASET_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Evaluate goal predictions in the original ego-frame coordinate units."""
    selected_device = torch.device(
        device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    model, stats = load_checkpoint(checkpoint_path, selected_device)
    dataloader = create_dataloader(
        dataset_dir=validation_dataset_dir,
        batch_size=batch_size,
        normalization_stats=stats,
    )

    absolute_errors = []
    squared_errors = []
    distances = []
    sample_count = 0

    for batch in dataloader:
        prediction = model(
            lidar=batch["lidar"].to(selected_device),
            ego_state=batch["ego_state"].to(selected_device),
            history_positions=batch["history_positions"].to(selected_device),
            history_speeds=batch["history_speeds"].to(selected_device),
            instruction=batch["instruction"].to(selected_device),
        )
        target = batch["target"].to(selected_device)

        prediction = (
            prediction * stats.target_std.to(selected_device)
            + stats.target_mean.to(selected_device)
        )
        target = (
            target * stats.target_std.to(selected_device)
            + stats.target_mean.to(selected_device)
        )
        error = prediction - target
        absolute_errors.append(error.abs().cpu())
        squared_errors.append(error.square().cpu())
        distances.append(error.norm(dim=1).cpu())
        sample_count += target.shape[0]

    if sample_count == 0:
        raise ValueError(
            f"No validation samples found in {validation_dataset_dir}"
        )

    absolute_error = torch.cat(absolute_errors)
    squared_error = torch.cat(squared_errors)
    distance = torch.cat(distances)
    metrics = {
        "samples": float(sample_count),
        "mae_x": absolute_error[:, 0].mean().item(),
        "mae_y": absolute_error[:, 1].mean().item(),
        "mae": absolute_error.mean().item(),
        "rmse": squared_error.mean().sqrt().item(),
        "mean_goal_distance_error": distance.mean().item(),
        "median_goal_distance_error": distance.median().item(),
    }
    return metrics


def _load_raw_sample(
    validation_dataset_dir: str | Path,
    sample_index: int,
) -> dict[str, Any]:
    """Load one deterministic raw record for visualization."""
    shard_paths = sorted(Path(validation_dataset_dir).glob("shard_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No JSONL shards found in: {validation_dataset_dir}"
        )

    remaining = sample_index
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                if remaining == 0:
                    return json.loads(line)
                remaining -= 1
    raise IndexError(
        f"Sample index {sample_index} out of range in {validation_dataset_dir}"
    )


def _load_instruction_samples(
    validation_dataset_dir: str | Path,
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Load raw validation records grouped by instruction and global index."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    sample_index = 0
    for shard_path in sorted(Path(validation_dataset_dir).glob("shard_*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                sample = json.loads(line)
                grouped[sample["instruction"]].append((sample_index, sample))
                sample_index += 1
    if not grouped:
        raise ValueError(f"No validation samples found in {validation_dataset_dir}")
    return dict(grouped)


def _random_instruction_samples(
    validation_dataset_dir: str | Path,
    seed: int | None = None,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Choose one reproducible-or-random record for every available instruction."""
    rng = random.Random(seed)
    grouped = _load_instruction_samples(validation_dataset_dir)
    return [
        (instruction, selected[0], selected[1])
        for instruction, samples in sorted(grouped.items())
        for selected in [rng.choice(samples)]
    ]


@torch.no_grad()
def predict_sample(
    sample: dict[str, Any],
    model: GoalPointPredictor,
    stats: NormalizationStats,
    device: torch.device,
    dataset_dir: str | Path = VALIDATION_DATASET_DIR,
) -> tuple[list[float], list[float]]:
    """Return predicted and target goal coordinates in the ego frame."""
    dataset = GoalPredictionDataset(
        dataset_dir=dataset_dir,
        normalization_stats=stats,
        shuffle_buffer_size=0,
    )
    converted = dataset._convert_sample(sample)
    prediction = model(
        lidar=converted["lidar"].unsqueeze(0).to(device),
        ego_state=converted["ego_state"].unsqueeze(0).to(device),
        history_positions=converted["history_positions"].unsqueeze(0).to(device),
        history_speeds=converted["history_speeds"].unsqueeze(0).to(device),
        instruction=converted["instruction"].unsqueeze(0).to(device),
    )
    prediction = (
        prediction * stats.target_std.to(device)
        + stats.target_mean.to(device)
    )[0].cpu()
    target = torch.tensor(
        sample["label"]["future_position_ego"], dtype=torch.float32
    )
    return prediction.tolist(), target.tolist()


def _add_3d_goal_markers(
    env: Any,
    predicted_goal_ego: list[float],
    target_goal_ego: list[float],
) -> list[tuple[str, tuple[float, float, float], tuple[int, int, int]]]:
    """Add predicted and target goal markers to MetaDrive's 3D scene."""
    from metadrive.engine.asset_loader import AssetLoader

    ego = env.agent
    ego_position = ego.position
    heading = float(ego.heading_theta)
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    marker_model = AssetLoader.file_path("models", "sphere.egg")
    marker_positions = []

    for name, goal, color in (
        ("predicted_goal", predicted_goal_ego, (0.1, 0.4, 1.0, 1.0)),
        ("target_goal", target_goal_ego, (1.0, 0.75, 0.05, 1.0)),
    ):
        forward, left = float(goal[0]), float(goal[1])
        world_x = ego_position[0] + forward * cos_heading - left * sin_heading
        world_y = ego_position[1] + forward * sin_heading + left * cos_heading
        marker = env.engine.loader.loadModel(marker_model)
        marker.setName(name)
        marker.setScale(2.5)
        marker.setColor(*color)
        marker.setPos(world_x, world_y, 3.0)
        marker.reparentTo(env.engine.render)
        marker_positions.append(
            (name, (world_x, world_y, 3.0), color[:3])
        )
    return marker_positions


def save_egocentric_view(
    sample: dict[str, Any],
    output_path: str | Path,
    predicted_goal_ego: list[float],
    target_goal_ego: list[float],
) -> None:
    """Replay MetaDrive and save its camera view with both goal markers."""
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
    except ImportError as error:
        raise RuntimeError(
            "MetaDrive is required for simulator visualization."
        ) from error

    meta = sample.get("meta", {})
    timestep = int(sample.get("timestep", 0))
    seed = int(str(sample.get("scenario_id", "seed_0")).split("seed_")[-1])
    map_spec = meta.get("map", "S")
    if isinstance(map_spec, str) and map_spec.isdigit():
        map_spec = int(map_spec)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = MetaDriveEnv(
        {
            "use_render": False,
            "image_observation": True,
            "window_size": (900, 600),
            "sensors": {"main_camera": ()},
            "vehicle_config": {"image_source": "main_camera"},
            "start_seed": seed,
            "map": map_spec,
            "traffic_density": meta.get("traffic_density", 0.1),
            "traffic_mode": meta.get("traffic_mode", "trigger"),
            "interface_panel": [],
            "log_level": 50,
        }
    )
    try:
        env.reset(seed=seed)
        action = sample["ego_state"].get("last_action", [0.0, 0.0])
        for _ in range(timestep):
            env.step(action)
        marker_positions = _add_3d_goal_markers(
            env, predicted_goal_ego, target_goal_ego
        )
        env.engine.render_frame()
        from panda3d.core import PNMImage

        screenshot = PNMImage()
        env.engine.win.getScreenshot(screenshot)
        image = np.empty(
            (screenshot.getYSize(), screenshot.getXSize(), 3),
            dtype=np.uint8,
        )
        for y in range(screenshot.getYSize()):
            for x in range(screenshot.getXSize()):
                pixel = screenshot.get_pixel(x, y)
                image[y, x] = [
                    int(pixel[channel]) for channel in range(3)
                ]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lens = env.engine.cam.node().getLens()
        from panda3d.core import Point2, Point3

        for _, (world_x, world_y, world_z), color in marker_positions:
            camera_point = env.engine.cam.getRelativePoint(
                env.engine.render, Point3(world_x, world_y, world_z)
            )
            projected = Point2()
            if lens.project(camera_point, projected):
                pixel_x = int((projected.x * 0.5 + 0.5) * image.shape[1])
                pixel_y = int(
                    (1.0 - (projected.y * 0.5 + 0.5)) * image.shape[0]
                )
                cv2.drawMarker(
                    image,
                    (pixel_x, pixel_y),
                    tuple(int(channel * 255) for channel in color),
                    cv2.MARKER_CROSS,
                    36,
                    5,
                )
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(
                f"Could not save MetaDrive camera image to "
                f"{output_path}"
            )
    finally:
        env.close()


def save_visualizations(
    validation_dataset_dir: str | Path = VALIDATION_DATASET_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    sample_index: int | None = None,
    output_dir: str | Path = PROJECT_DIR / "validation_visualizations",
    device: str | torch.device | None = None,
    random_seed: int | None = None,
) -> list[tuple[Path, Path]]:
    """Save BEV and egocentric images for one random sample per instruction."""
    selected_device = torch.device(
        device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    model, stats = load_checkpoint(checkpoint_path, selected_device)
    if sample_index is None:
        selected_samples = _random_instruction_samples(
            validation_dataset_dir, random_seed
        )
    else:
        sample = _load_raw_sample(validation_dataset_dir, sample_index)
        selected_samples = [
            (sample["instruction"], sample_index, sample)
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from visualize_bev import add_hud, render_metadrive_bev
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "BEV visualization requires pygame and opencv-python."
        ) from error

    outputs = []
    for instruction, selected_index, sample in selected_samples:
        prediction, target = predict_sample(
            sample, model, stats, selected_device, validation_dataset_dir
        )
        stem = f"sample_{selected_index:05d}_{instruction}"
        bev_path = output_path / f"{stem}_bev.png"
        ego_path = output_path / f"{stem}_egocentric.png"
        bev = render_metadrive_bev(sample, predicted_goal_ego=prediction)
        cv2.imwrite(str(bev_path), add_hud(bev, sample))
        save_egocentric_view(sample, ego_path, prediction, target)
        error_distance = torch.tensor(prediction).sub(torch.tensor(target)).norm()
        print(
            f"{instruction}: sample {selected_index} | "
            f"goal error {error_distance.item():.3f} m\n"
            f"  saved BEV: {bev_path}\n"
            f"  saved egocentric view: {ego_path}"
        )
        outputs.append((bev_path, ego_path))
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained goal predictor.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Visualize one record instead of one random record per instruction.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for repeatable instruction-wise sampling.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "validation_visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.visualize:
        save_visualizations(
            sample_index=args.sample_index,
            random_seed=args.random_seed,
            output_dir=args.output_dir,
        )
        raise SystemExit(0)

    results = evaluate()
    print("validation metrics:")
    for name, value in results.items():
        if name == "samples":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.4f}")
