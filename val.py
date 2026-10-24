from pathlib import Path
from typing import Any

import torch
from torch import nn

from train import (
    MODEL_PATH,
    GoalPointPredictor,
    NormalizationStats,
    create_dataloader,
)


PROJECT_DIR = Path(__file__).parent
VALIDATION_DATASET_DIR = PROJECT_DIR / "goal_dataset" / "val"


def load_checkpoint(
    checkpoint_path: str | Path = MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[GoalPointPredictor, NormalizationStats]:
    """Load the trained model and the training normalization statistics."""
    path = Path(checkpoint_path)
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
    checkpoint_path: str | Path = MODEL_PATH,
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


if __name__ == "__main__":
    results = evaluate()
    print("validation metrics:")
    for name, value in results.items():
        if name == "samples":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.4f}")
