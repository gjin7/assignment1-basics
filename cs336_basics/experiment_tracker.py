from pathlib import Path
import time
import json
from dataclasses import asdict

from cs336_basics.config import Config

class ExperimentTracker:
    """
    Simple experiment tracker that logs: 
    - global step
    - wall-clock time
    """

    def __init__(
        self, 
        run_dir: str | Path, 
        config: Config,
        wall_time_offset: float = 0.0,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.start_time = time.time()
        self.wall_time_offset = wall_time_offset

        with self.config_path.open("w") as f:
            json.dump(
                {
                    "config": asdict(config),
                    "run_dir": str(self.run_dir),
                    "wall_time_offset": self.wall_time_offset,
                },
                f,
                indent=2,
            )

    def elapsed_time(self) -> float:
        return self.wall_time_offset + time.time() - self.start_time

    def log(self, step: int, metrics: dict[str, float]) -> None:
        row = {
            "step": step, 
            "wall_time": self.elapsed_time(),
            **metrics, 
        }

        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
