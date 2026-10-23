from pathlib import Path
import json
from typing import Dict, Iterator, Any

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


DATASET_DIR = Path(__file__).parent / "goal_dataset" / "train"

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


class GoalPredictionDataset(IterableDataset):
    """
    Streams goal-prediction samples from JSONL dataset shards.

    Each sample contains:

        inputs:
            ego_state:       [11]
            lidar:            [240]
            instruction_id:  scalar integer
            history_positions:[3, 2]
            history_speeds:   [3]
            history_lidar:    [3, 240]

        target:
            future_position_ego: [2]

    The target is the ego-frame position of the vehicle 5 seconds
    into the future.
    """

    def __init__(self, dataset_dir: str | Path = DATASET_DIR):
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

    def _convert_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        history = self._get_history(sample)

        label = sample["label"]["future_position_ego"]

        if len(label) != 2:
            raise ValueError(
                f"Expected a two-dimensional target coordinate, "
                f"but received {len(label)} values"
            )

        return {
            "ego_state": self._get_ego_state(sample),
            "lidar": self._get_lidar(sample),
            "instruction": self._get_instruction_id(sample),
            "history_positions": history["positions"],
            "history_speeds": history["speeds"],
            "history_lidar": history["lidar"],
            "target": torch.tensor(label, dtype=torch.float32),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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


def create_dataloader(
    dataset_dir: str | Path = DATASET_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GoalPredictionDataset(dataset_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )