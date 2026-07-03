"""Sweep replay_eval across pick_place_bin_v1 checkpoints on the full eval split.

Loads each checkpoint once, evaluates every episode in the eval dataset, and
reports a single pooled per-joint RMSE / gripper accuracy / latency per
checkpoint so the training-step trajectory is directly comparable.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from replay_eval import _eval_episode

CONFIG_NAME = "pi05_ur5e_avea_lora"
CKPT_ROOT = pathlib.Path("/home/stallion/openpi/checkpoints/pi05_ur5e_avea_lora/pick_place_bin_v1")
STEPS = [5000, 10000, 15000, 19999]
DATASET_ROOT = pathlib.Path(
    "/home/stallion/datasets/lerobot/relling/ur5e-avea-teleop-v0-eval-20260520/data/chunk-000"
)
PROMPT = "pick and place the object into the bin"
STRIDE = 10
MAX_SAMPLES = 10_000
OUT = pathlib.Path("/home/stallion/openpi/logs/eval_sweep.json")


def _pool(results: list[dict]) -> dict:
    weights = np.array([r["samples"] for r in results], dtype=np.float64)
    per_joint = np.stack([r["per_joint_mse"] for r in results])
    pooled_joint_mse = np.average(per_joint, axis=0, weights=weights)
    gripper_acc = float(np.average([r["gripper_match"] for r in results], weights=weights))
    return {
        "episodes": len(results),
        "total_samples": int(weights.sum()),
        "per_joint_rmse_rad": np.sqrt(pooled_joint_mse).tolist(),
        "mean_joint_rmse_rad": float(np.sqrt(pooled_joint_mse.mean())),
        "j5_rmse_rad": float(np.sqrt(pooled_joint_mse[5])),
        "gripper_accuracy": gripper_acc,
        "nonfinite": int(sum(r["nonfinite"] for r in results)),
        "max_abs_delta_rad": float(max(r["max_abs_delta"] for r in results)),
        "warm_infer_ms_p50": float(np.median([r["warm_call_ms_p50"] for r in results])),
    }


def main() -> int:
    train_config = _config.get_config(CONFIG_NAME)
    episodes = sorted(p.name for p in DATASET_ROOT.glob("episode_*.parquet"))
    summary: dict[str, dict] = {}

    for step in STEPS:
        ckpt = CKPT_ROOT / str(step)
        print(f"\n==== loading checkpoint {step} ====", flush=True)
        policy = _policy_config.create_trained_policy(train_config, ckpt, default_prompt=PROMPT)
        results = [
            _eval_episode(
                policy,
                DATASET_ROOT / ep,
                prompt=PROMPT,
                stride=STRIDE,
                max_samples=MAX_SAMPLES,
            )
            for ep in episodes
        ]
        summary[str(step)] = _pool(results)
        print(f"step {step}: {json.dumps(summary[str(step)], indent=2)}", flush=True)

    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 78)
    print(f"{'step':>6} {'mean_rmse':>10} {'j0':>7} {'j1':>7} {'j2':>7} {'j3':>7} {'j4':>7} {'j5':>7} {'grip%':>7} {'ms':>5}")
    for step in STEPS:
        s = summary[str(step)]
        j = s["per_joint_rmse_rad"]
        print(
            f"{step:>6} {s['mean_joint_rmse_rad']:>10.4f} "
            + " ".join(f"{v:>7.4f}" for v in j)
            + f" {s['gripper_accuracy']*100:>6.1f} {s['warm_infer_ms_p50']:>5.0f}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
