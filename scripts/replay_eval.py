"""Offline replay evaluation for a trained ur5e openpi policy.

Walks a LeRobot parquet episode, calls ``policy.infer`` per sampled frame, and
compares the predicted 16-step action chunk against the ground-truth next 16
absolute actions. Reports per-joint MSE, MSE-vs-horizon, gripper accuracy,
and out-of-bounds counts.
"""

from __future__ import annotations

import argparse
import io
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "pick and place the object into the bin"
HORIZON = 16


def _decode_image(value) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(value))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _build_observation(row: pd.Series, prompt: str) -> dict:
    return {
        "base_rgb": _decode_image(row["base_rgb"]),
        "wrist_rgb": _decode_image(row["wrist_rgb"]),
        "overhead_rgb": _decode_image(row["overhead_rgb"]),
        "joints": np.asarray(row["joints"], dtype=np.float32),
        "joints_velocity": np.asarray(row["joints_velocity"], dtype=np.float32),
        "gripper": np.asarray(row["gripper"], dtype=np.float32).reshape(-1),
        "prompt": prompt,
    }


def _eval_episode(
    policy,
    parquet_path: pathlib.Path,
    *,
    prompt: str,
    stride: int,
    max_samples: int,
) -> dict:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    if n <= HORIZON:
        raise ValueError(f"episode {parquet_path.name} too short ({n} frames) for horizon {HORIZON}")

    indices = list(range(0, n - HORIZON, stride))[:max_samples]
    logger.info("episode=%s frames=%d sampling %d frames stride=%d", parquet_path.name, n, len(indices), stride)

    pred_chunks = []
    gt_chunks = []
    timings = []

    for t in indices:
        obs = _build_observation(df.iloc[t], prompt=prompt)
        out = policy.infer(obs)
        pred = np.asarray(out["actions"], dtype=np.float32)
        if pred.shape != (HORIZON, 7):
            raise ValueError(f"unexpected action shape {pred.shape} at frame {t}")
        gt_actions = np.stack([np.asarray(df.iloc[t + k]["actions"], dtype=np.float32) for k in range(HORIZON)])
        pred_chunks.append(pred)
        gt_chunks.append(gt_actions)
        timings.append(float(out.get("policy_timing", {}).get("infer_ms", 0.0)))

    pred_arr = np.stack(pred_chunks)
    gt_arr = np.stack(gt_chunks)
    diff = pred_arr - gt_arr

    per_joint_mse = (diff[..., :6] ** 2).mean(axis=(0, 1))
    horizon_mse_joints = (diff[..., :6] ** 2).mean(axis=(0, 2))
    horizon_mse_gripper = (diff[..., 6] ** 2).mean(axis=0)
    gripper_pred_binary = (pred_arr[..., 6] > 0.5).astype(np.int32)
    gripper_gt_binary = (gt_arr[..., 6] > 0.5).astype(np.int32)
    gripper_match = (gripper_pred_binary == gripper_gt_binary).mean()

    nonfinite = int((~np.isfinite(pred_arr)).sum())
    max_abs_delta = float(np.abs(diff[..., :6]).max())
    max_abs_pred_joint = float(np.abs(pred_arr[..., :6]).max())

    return {
        "episode": parquet_path.name,
        "samples": len(indices),
        "per_joint_mse": per_joint_mse,
        "horizon_mse_joints": horizon_mse_joints,
        "horizon_mse_gripper": horizon_mse_gripper,
        "gripper_match": float(gripper_match),
        "nonfinite": nonfinite,
        "max_abs_delta": max_abs_delta,
        "max_abs_pred_joint": max_abs_pred_joint,
        "first_call_ms": timings[0] if timings else 0.0,
        "warm_call_ms_p50": float(np.median(timings[1:])) if len(timings) > 1 else 0.0,
    }


def _print_report(results: list[dict]) -> None:
    print()
    print("=" * 78)
    for r in results:
        print(f"episode: {r['episode']}  samples={r['samples']}")
        print("  per-joint MSE (rad^2):")
        for k, v in enumerate(r["per_joint_mse"]):
            print(f"    joint_{k}: {v:.6f}  (rmse {np.sqrt(v):.4f} rad ~= {np.degrees(np.sqrt(v)):.2f} deg)")
        print("  horizon-step joint MSE:")
        for k in (0, 1, 3, 7, 15):
            if k < len(r["horizon_mse_joints"]):
                print(f"    step {k:2d}: joints={r['horizon_mse_joints'][k]:.6f}  gripper={r['horizon_mse_gripper'][k]:.6f}")
        print(f"  gripper binary accuracy: {r['gripper_match']*100:.1f}%")
        print(f"  non-finite outputs: {r['nonfinite']}")
        print(f"  max |pred - gt| joint delta: {r['max_abs_delta']:.4f} rad")
        print(f"  max |pred| joint:           {r['max_abs_pred_joint']:.4f} rad")
        print(f"  infer ms: first={r['first_call_ms']:.0f}  warm-median={r['warm_call_ms_p50']:.0f}")
        print("-" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_ur5e_avea_lora")
    parser.add_argument(
        "--ckpt-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/stallion/openpi/checkpoints/pi05_ur5e_avea_lora/pick_place_bin_v1/19999"),
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/stallion/datasets/lerobot/relling/ur5e-avea-teleop-v0/data/chunk-000"),
    )
    parser.add_argument("--episodes", nargs="*", default=["episode_000000.parquet", "episode_000012.parquet"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stride", type=int, default=30, help="sample every N frames")
    parser.add_argument("--max-samples", type=int, default=12, help="cap samples per episode")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname).1s] %(message)s")

    logger.info("loading config %s", args.config_name)
    train_config = _config.get_config(args.config_name)
    logger.info("loading policy from %s", args.ckpt_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.ckpt_dir,
        default_prompt=args.prompt,
    )

    results = []
    for episode in args.episodes:
        parquet_path = args.dataset_root / episode
        if not parquet_path.exists():
            logger.warning("missing %s, skipping", parquet_path)
            continue
        results.append(
            _eval_episode(
                policy,
                parquet_path,
                prompt=args.prompt,
                stride=args.stride,
                max_samples=args.max_samples,
            )
        )

    _print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
