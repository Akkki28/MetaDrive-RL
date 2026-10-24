from pathlib import Path
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterator, Any

import torch
from torch import Tensor, nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


DATASET_DIR = Path(__file__).parent / "goal_dataset" / "train"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "goal_point_predictor.pt"

NUM_LIDAR_POINTS = 240
HISTORY_LENGTH = 3
NUM_INSTRUCTIONS = 7


INSTRUCTION_TO_ID = {
    "go_straight": 0,
    "go_straight_fast": 1,
    "go_straight_slow": 2,
    "turn_left": 3,
    "turn_right": 4,
    "lane_change_left": 5,
    "lane_change_right": 6,
}


class GoalPointPredictor(nn.Module):
    """Predict a future ego-frame goal from lidar, ego state, and an instruction."""

    def __init__(
        self,
        ego_state_dim: int = 12,
        history_length: int = HISTORY_LENGTH,
        instruction_count: int = NUM_INSTRUCTIONS,
        model_dim: int = 128,
        instruction_dim: int = 32,
        attention_heads: int = 4,
    ):
        super().__init__()

        if model_dim % attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")

        self.lidar_token = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.ego_token = nn.Sequential(
            nn.Linear(ego_state_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.history_token = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

        self.lidar_to_ego_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.instruction_embedding = nn.Embedding(
            instruction_count,
            instruction_dim,
        )
        self.goal_head = nn.Sequential(
            nn.Linear(model_dim + instruction_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
        )

        self.history_length = history_length

    def forward(
        self,
        lidar: Tensor,
        ego_state: Tensor,
        history_positions: Tensor,
        history_speeds: Tensor,
        instruction: Tensor,
    ) -> Tensor:
        """Return normalized future ego-frame coordinates with shape [batch, 2]."""
        if history_positions.shape[1] != self.history_length:
            raise ValueError(
                f"Expected {self.history_length} history steps, "
                f"but received {history_positions.shape[1]}"
            )

        lidar_tokens = self.lidar_token(lidar.unsqueeze(-1))
        history_features = torch.cat(
            (history_positions, history_speeds.unsqueeze(-1)),
            dim=-1,
        )
        ego_tokens = torch.cat(
            (
                self.ego_token(ego_state).unsqueeze(1),
                self.history_token(history_features),
            ),
            dim=1,
        )

        attended_ego, _ = self.lidar_to_ego_attention(
            query=ego_tokens,
            key=lidar_tokens,
            value=lidar_tokens,
        )
        fused_ego = self.attention_norm(ego_tokens + attended_ego).mean(dim=1)
        instruction_features = self.instruction_embedding(instruction)
        return self.goal_head(
            torch.cat((fused_ego, instruction_features), dim=-1)
        )


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature statistics fitted on the training split."""

    ego_state_mean: torch.Tensor
    ego_state_std: torch.Tensor
    history_positions_mean: torch.Tensor
    history_positions_std: torch.Tensor
    history_speeds_mean: torch.Tensor
    history_speeds_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor


class GoalPredictionDataset(IterableDataset):
    """
    Streams goal-prediction samples from JSONL dataset shards.

    Each sample contains:

        inputs:
            ego_state:       [12]
            lidar:            [240]
            instruction_id:  scalar integer
            history_positions:[3, 2]
            history_speeds:   [3]
            history_lidar:    [3, 240] (loaded for compatibility, not used)

        target:
            future_position_ego: [2]

    The target is the ego-frame position of the vehicle 5 seconds
    into the future.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DATASET_DIR,
        normalization_stats: NormalizationStats | None = None,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ):
        super().__init__()

        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir}"
            )

        self.shard_paths = sorted(self.dataset_dir.glob("shard_*.jsonl"))

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No JSONL shards found in: {self.dataset_dir}"
            )

        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

        self.normalization_stats = normalization_stats
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed


    def _get_ego_state(self, sample: Dict[str, Any]) -> torch.Tensor:
        ego_state = sample["ego_state"]

        values = [
            ego_state["speed"],
            ego_state["steering"],
            ego_state["yaw_rate"],
            ego_state["heading"],
            ego_state["position"][0],
            ego_state["position"][1],
            ego_state["velocity"][0],
            ego_state["velocity"][1],
            ego_state["last_action"][0],
            ego_state["last_action"][1],
            ego_state["lateral_offset"],
            ego_state["heading_diff"],
        ]

        return torch.tensor(values, dtype=torch.float32)

    def _get_instruction_id(self, sample: Dict[str, Any]) -> torch.Tensor:
        instruction = sample["instruction"]

        if instruction not in INSTRUCTION_TO_ID:
            raise ValueError(f"Unknown instruction: {instruction}")

        return torch.tensor(
            INSTRUCTION_TO_ID[instruction],
            dtype=torch.long,
        )

    def _get_lidar(self, sample: Dict[str, Any]) -> torch.Tensor:
        lidar = sample["lidar"]["cloud_points"]

        if len(lidar) != NUM_LIDAR_POINTS:
            raise ValueError(
                f"Expected {NUM_LIDAR_POINTS} lidar values, "
                f"but received {len(lidar)}"
            )

        return torch.tensor(lidar, dtype=torch.float32)

    def _get_history(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        history = sample["history"]

        positions = history["ego_positions"]
        speeds = history["ego_speeds"]
        lidar = history["lidar_cloud_points"]

        if len(positions) != HISTORY_LENGTH:
            raise ValueError(
                f"Expected {HISTORY_LENGTH} historical positions, "
                f"but received {len(positions)}"
            )

        if len(speeds) != HISTORY_LENGTH:
            raise ValueError(
                f"Expected {HISTORY_LENGTH} historical speeds, "
                f"but received {len(speeds)}"
            )

        if len(lidar) != HISTORY_LENGTH:
            raise ValueError(
                f"Expected {HISTORY_LENGTH} historical lidar frames, "
                f"but received {len(lidar)}"
            )

        for frame in lidar:
            if len(frame) != NUM_LIDAR_POINTS:
                raise ValueError(
                    f"Expected {NUM_LIDAR_POINTS} values in every historical "
                    f"lidar frame, but received {len(frame)}"
                )

        return {
            "positions": torch.tensor(positions, dtype=torch.float32),
            "speeds": torch.tensor(speeds, dtype=torch.float32),
            "lidar": torch.tensor(lidar, dtype=torch.float32),
        }

    def _convert_sample(
        self,
        sample: Dict[str, Any],
        normalize: bool = True,
    ) -> Dict[str, Any]:
        history = self._get_history(sample)

        label = sample["label"]["future_position_ego"]

        if len(label) != 2:
            raise ValueError(
                f"Expected a two-dimensional target coordinate, "
                f"but received {len(label)} values"
            )

        converted = {
            "ego_state": self._get_ego_state(sample),
            "lidar": self._get_lidar(sample),
            "instruction": self._get_instruction_id(sample),
            "history_positions": history["positions"],
            "history_speeds": history["speeds"],
            "history_lidar": history["lidar"],
            "target": torch.tensor(label, dtype=torch.float32),
        }

        if normalize and self.normalization_stats is not None:
            stats = self.normalization_stats
            converted["ego_state"] = (
                converted["ego_state"] - stats.ego_state_mean
            ) / stats.ego_state_std
            converted["history_positions"] = (
                converted["history_positions"]
                - stats.history_positions_mean
            ) / stats.history_positions_std
            converted["history_speeds"] = (
                converted["history_speeds"] - stats.history_speeds_mean
            ) / stats.history_speeds_std
            converted["target"] = (
                converted["target"] - stats.target_mean
            ) / stats.target_std

        return converted

    def _iter_samples(self) -> Iterator[Dict[str, Any]]:
        worker_info = get_worker_info()

        if worker_info is None:
            shard_paths = self.shard_paths
        else:
            worker_id = worker_info.id
            number_of_workers = worker_info.num_workers

            shard_paths = self.shard_paths[
                worker_id::number_of_workers
            ]

        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {shard_path}, "
                            f"line {line_number}"
                        ) from error

                    yield self._convert_sample(sample)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        samples = self._iter_samples()

        if self.shuffle_buffer_size == 0:
            yield from samples
            return

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = random.Random(self.seed + worker_id)
        buffer = []

        for sample in samples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue

            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample

        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


def fit_normalization_stats(
    dataset_dir: str | Path = DATASET_DIR,
) -> NormalizationStats:
    """Fit continuous-feature statistics from a training dataset stream."""

    dataset = GoalPredictionDataset(
        dataset_dir,
        shuffle_buffer_size=0,
    )
    sums = {
        "ego_state": torch.zeros(12, dtype=torch.float64),
        "history_positions": torch.zeros(2, dtype=torch.float64),
        "history_speeds": torch.zeros(1, dtype=torch.float64),
        "target": torch.zeros(2, dtype=torch.float64),
    }
    squared_sums = {name: values.clone() for name, values in sums.items()}
    counts = {name: 0 for name in sums}

    for sample in dataset._iter_samples():
        values = {
            "ego_state": sample["ego_state"].double(),
            "history_positions": sample["history_positions"].double().reshape(-1, 2),
            "history_speeds": sample["history_speeds"].double().reshape(-1, 1),
            "target": sample["target"].double().reshape(-1, 2),
        }

        for name, tensor in values.items():
            sums[name] += tensor.sum(dim=0)
            squared_sums[name] += (tensor * tensor).sum(dim=0)
            counts[name] += tensor.shape[0]

    if counts["ego_state"] == 0:
        raise ValueError(f"No samples found in: {dataset.dataset_dir}")

    stats = {}
    for name in sums:
        mean = sums[name] / counts[name]
        variance = squared_sums[name] / counts[name] - mean.square()
        stats[f"{name}_mean"] = mean.float()
        stats[f"{name}_std"] = variance.clamp_min(1e-12).sqrt().float()

    return NormalizationStats(**stats)


def create_dataloader(
    dataset_dir: str | Path = DATASET_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
    normalization_stats: NormalizationStats | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> DataLoader:
    if normalization_stats is None:
        normalization_stats = fit_normalization_stats(dataset_dir)

    dataset = GoalPredictionDataset(
        dataset_dir,
        normalization_stats=normalization_stats,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: GoalPointPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one supervised training pass and return the mean normalized loss."""
    model.train()
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    batch_count = 0

    for batch in dataloader:
        lidar = batch["lidar"].to(device)
        ego_state = batch["ego_state"].to(device)
        history_positions = batch["history_positions"].to(device)
        history_speeds = batch["history_speeds"].to(device)
        instruction = batch["instruction"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            lidar=lidar,
            ego_state=ego_state,
            history_positions=history_positions,
            history_speeds=history_speeds,
            instruction=instruction,
        )
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError("The DataLoader produced no batches")

    return total_loss / batch_count


def train_model(
    dataset_dir: str | Path = DATASET_DIR,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
) -> GoalPointPredictor:
    """Build the DataLoader and train a goal predictor for a few epochs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    selected_device = torch.device(
        device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    normalization_stats = fit_normalization_stats(dataset_dir)
    dataloader = create_dataloader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        normalization_stats=normalization_stats,
    )
    model = GoalPointPredictor().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=selected_device,
        )
        print(f"epoch {epoch + 1}/{epochs} - loss: {loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "ego_state_dim": 12,
                "history_length": HISTORY_LENGTH,
                "instruction_count": NUM_INSTRUCTIONS,
            },
            "normalization_stats": {
                name: getattr(normalization_stats, name)
                for name in NormalizationStats.__dataclass_fields__
            },
        },
        MODEL_PATH,
    )
    print(f"saved model checkpoint to {MODEL_PATH}")

    return model
