"""Hotspot review queue generator for a trained ur5e policy.

Runs the policy on parquet episodes (same path as replay_eval), identifies
frames where predictions diverge most from teleop GT under several criteria,
and renders a PNG contact sheet per hotspot for human inspection. Also dumps
a queue.json index summarizing each hotspot and its reason for inclusion.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "pick and place the object into the bin"
HORIZON = 16
JOINT_LABELS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


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


@dataclass
class Sample:
    episode_path: pathlib.Path
    episode_name: str
    frame_index: int
    sample_idx: int
    pred: np.ndarray
    gt: np.ndarray
    images: dict
    scores: dict = field(default_factory=dict)


def _score(sample: Sample) -> dict:
    diff = sample.pred - sample.gt
    joint_diff = diff[:, :6]
    per_joint_rmse = np.sqrt((joint_diff ** 2).mean(axis=0))
    return {
        "total_joint_mse": float((joint_diff ** 2).mean()),
        "max_abs_joint_delta": float(np.abs(joint_diff).max()),
        "per_joint_rmse": per_joint_rmse.tolist(),
        "j5_rmse": float(per_joint_rmse[5]),
        "j2_rmse": float(per_joint_rmse[1]),
        "elbow_rmse": float(per_joint_rmse[2]),
        "gripper_disagree_steps": int(
            ((sample.pred[:, 6] > 0.5) != (sample.gt[:, 6] > 0.5)).sum()
        ),
        "gripper_value_rmse": float(np.sqrt(((sample.pred[:, 6] - sample.gt[:, 6]) ** 2).mean())),
    }


def _collect_samples(
    policy,
    parquet_path: pathlib.Path,
    *,
    prompt: str,
    stride: int,
    max_samples: int,
) -> list[Sample]:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    if n <= HORIZON:
        logger.warning("episode %s too short (%d), skipping", parquet_path.name, n)
        return []
    indices = list(range(0, n - HORIZON, stride))[:max_samples]
    logger.info("episode=%s frames=%d sampling %d", parquet_path.name, n, len(indices))

    samples = []
    for sample_idx, t in enumerate(indices):
        obs = _build_observation(df.iloc[t], prompt=prompt)
        out = policy.infer(obs)
        pred = np.asarray(out["actions"], dtype=np.float32)
        if pred.shape != (HORIZON, 7):
            raise ValueError(f"unexpected action shape {pred.shape} at frame {t}")
        gt = np.stack([np.asarray(df.iloc[t + k]["actions"], dtype=np.float32) for k in range(HORIZON)])
        images = {
            "base_rgb": _decode_image(df.iloc[t]["base_rgb"]),
            "wrist_rgb": _decode_image(df.iloc[t]["wrist_rgb"]),
            "overhead_rgb": _decode_image(df.iloc[t]["overhead_rgb"]),
        }
        s = Sample(
            episode_path=parquet_path,
            episode_name=parquet_path.stem,
            frame_index=int(t),
            sample_idx=sample_idx,
            pred=pred,
            gt=gt,
            images=images,
        )
        s.scores = _score(s)
        samples.append(s)
    return samples


CRITERIA = [
    ("total_horizon_mse", "total_joint_mse", "highest aggregate joint MSE across 16-step horizon"),
    ("max_single_step_delta", "max_abs_joint_delta", "worst single-step joint deviation (rad)"),
    ("j5_wrist_outlier", "j5_rmse", "wrist (J5) RMSE outlier"),
    ("j2_shoulder_outlier", "j2_rmse", "shoulder-lift (J2) RMSE outlier"),
    ("gripper_binary_mismatch", "gripper_disagree_steps", "steps where gripper binary class flips"),
    ("gripper_value_drift", "gripper_value_rmse", "gripper value RMSE even if binary agrees"),
]


def _rank_hotspots(samples: list[Sample], per_criterion: int) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for crit_name, score_key, blurb in CRITERIA:
        ranked = sorted(samples, key=lambda s: s.scores[score_key], reverse=True)
        for s in ranked[:per_criterion]:
            if score_key == "gripper_disagree_steps" and s.scores[score_key] == 0:
                continue
            key = (s.episode_name, s.frame_index)
            entry = by_key.setdefault(
                key,
                {"sample": s, "reasons": [], "primary_score": s.scores[score_key], "primary_crit": crit_name},
            )
            entry["reasons"].append(
                {"criterion": crit_name, "score": float(s.scores[score_key]), "blurb": blurb}
            )
    hotspots = list(by_key.values())
    hotspots.sort(key=lambda h: h["sample"].scores["total_joint_mse"], reverse=True)
    return hotspots


def _render_contact_sheet(hotspot: dict, out_path: pathlib.Path) -> None:
    s: Sample = hotspot["sample"]
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.2, 1.0, 1.0, 1.0, 0.6])

    for col, key in enumerate(("base_rgb", "wrist_rgb", "overhead_rgb")):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(s.images[key])
        ax.set_title(f"{key} @ frame {s.frame_index}", fontsize=10)
        ax.axis("off")

    steps = np.arange(HORIZON)
    plot_specs = [
        (1, 0, 0, "J0 shoulder_pan (rad)"),
        (1, 1, 1, "J1 shoulder_lift (rad)"),
        (1, 2, 2, "J2 elbow (rad)"),
        (2, 0, 3, "J3 wrist_1 (rad)"),
        (2, 1, 4, "J4 wrist_2 (rad)"),
        (2, 2, 5, "J5 wrist_3 (rad)"),
        (3, 0, 6, "gripper"),
    ]
    for row, col, action_idx, label in plot_specs:
        ax = fig.add_subplot(gs[row, col])
        ax.plot(steps, s.gt[:, action_idx], "o-", color="tab:blue", label="GT", markersize=3)
        ax.plot(steps, s.pred[:, action_idx], "x--", color="tab:red", label="pred", markersize=4)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("horizon step", fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    ax_diff = fig.add_subplot(gs[3, 1])
    for j in range(6):
        ax_diff.plot(steps, s.pred[:, j] - s.gt[:, j], label=f"J{j}", linewidth=1)
    ax_diff.axhline(0, color="black", linewidth=0.5)
    ax_diff.set_title("pred - GT per joint (rad)", fontsize=9)
    ax_diff.set_xlabel("horizon step", fontsize=8)
    ax_diff.legend(fontsize=7, ncol=2, loc="best")
    ax_diff.grid(alpha=0.3)
    ax_diff.tick_params(labelsize=7)

    ax_info = fig.add_subplot(gs[3, 2])
    ax_info.axis("off")
    info_lines = [
        f"episode: {s.episode_name}",
        f"frame: {s.frame_index}  (sample #{s.sample_idx})",
        "",
        "scores:",
        f"  total joint MSE: {s.scores['total_joint_mse']:.6f}",
        f"  max |Δ|:         {s.scores['max_abs_joint_delta']:.4f} rad",
        f"  per-joint RMSE:",
    ]
    for j, rmse in enumerate(s.scores["per_joint_rmse"]):
        info_lines.append(f"    J{j}: {np.degrees(rmse):.2f}°")
    info_lines.extend(
        [
            f"  gripper Δ steps: {s.scores['gripper_disagree_steps']}",
            f"  gripper value RMSE: {s.scores['gripper_value_rmse']:.4f}",
        ]
    )
    ax_info.text(0, 1, "\n".join(info_lines), fontsize=9, family="monospace", va="top")

    ax_reasons = fig.add_subplot(gs[4, :])
    ax_reasons.axis("off")
    reason_text = "FLAGGED BECAUSE:\n" + "\n".join(
        f"  · {r['criterion']:24s} {r['score']:.4f}  — {r['blurb']}" for r in hotspot["reasons"]
    )
    ax_reasons.text(
        0, 1, reason_text, fontsize=10, family="monospace", va="top",
        bbox={"facecolor": "#fff7e6", "edgecolor": "#e8a93f", "boxstyle": "round,pad=0.4"},
    )

    fig.suptitle(
        f"REVIEW HOTSPOT  ·  {s.episode_name} @ frame {s.frame_index}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_ur5e_avea_lora")
    parser.add_argument(
        "--ckpt-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/stallion/openpi/checkpoints/pi05_ur5e_avea_lora/pick_place_bin_v1/19999"),
    )
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--per-criterion", type=int, default=3, help="top-K samples per ranking criterion")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="output directory; defaults to {ckpt-dir}/review/{timestamp}",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname).1s] %(message)s")

    out_dir = args.out_dir
    if out_dir is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.ckpt_dir / "review" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)

    manifest_path = args.dataset_root.parent.parent / "ur5e_manifest.json"
    episode_index_map: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for m in manifest:
            ep_name = f"episode_{m['episode_index']:06d}"
            episode_index_map[ep_name] = {
                "session_id": m["session_id"],
                "real_episode_id": m["episode_id"],
                "source_path": m.get("source_path", ""),
                "mcap_path": m.get("metadata", {}).get("mcap_path", ""),
            }
        logger.info("loaded manifest with %d episodes", len(episode_index_map))
    else:
        logger.warning("ur5e_manifest.json not found at %s; verdicts.template.json will have empty session_id/episode_id", manifest_path)

    logger.info("loading config %s", args.config_name)
    train_config = _config.get_config(args.config_name)
    logger.info("loading policy from %s", args.ckpt_dir)
    policy = _policy_config.create_trained_policy(
        train_config, args.ckpt_dir, default_prompt=args.prompt
    )

    all_samples = []
    for episode in args.episodes:
        parquet_path = args.dataset_root / episode
        if not parquet_path.exists():
            logger.warning("missing %s, skipping", parquet_path)
            continue
        all_samples.extend(
            _collect_samples(
                policy, parquet_path,
                prompt=args.prompt, stride=args.stride, max_samples=args.max_samples,
            )
        )
    logger.info("collected %d total samples", len(all_samples))
    if not all_samples:
        logger.error("no samples; aborting")
        return 1

    hotspots = _rank_hotspots(all_samples, per_criterion=args.per_criterion)
    logger.info("ranked %d unique hotspots from %d criteria", len(hotspots), len(CRITERIA))

    queue_entries = []
    verdict_entries = []
    for rank, h in enumerate(hotspots):
        s: Sample = h["sample"]
        fname = f"{rank:02d}_{s.episode_name}_f{s.frame_index:05d}.png"
        out_path = out_dir / fname
        _render_contact_sheet(h, out_path)
        provenance = episode_index_map.get(s.episode_name, {})
        queue_entries.append(
            {
                "rank": rank,
                "png": fname,
                "episode": s.episode_name,
                "session_id": provenance.get("session_id", ""),
                "real_episode_id": provenance.get("real_episode_id", ""),
                "mcap_path": provenance.get("mcap_path", ""),
                "frame_index": s.frame_index,
                "scores": s.scores,
                "reasons": h["reasons"],
            }
        )
        verdict_entries.append(
            {
                "hotspot_rank": rank,
                "png": fname,
                "episode": s.episode_name,
                "session_id": provenance.get("session_id", ""),
                "real_episode_id": provenance.get("real_episode_id", ""),
                "frame_index": s.frame_index,
                "action": "TODO",
                "notes": "",
            }
        )
        logger.info("[%02d] %s f%d -> %s", rank, s.episode_name, s.frame_index, fname)

    queue_path = out_dir / "queue.json"
    queue_path.write_text(json.dumps(
        {
            "ckpt_dir": str(args.ckpt_dir),
            "dataset_root": str(args.dataset_root),
            "config_name": args.config_name,
            "prompt": args.prompt,
            "stride": args.stride,
            "max_samples": args.max_samples,
            "per_criterion": args.per_criterion,
            "hotspots": queue_entries,
        },
        indent=2,
    ))

    verdicts_template_path = out_dir / "verdicts.template.json"
    verdicts_template_path.write_text(json.dumps(
        {
            "_instructions": (
                "Fill `action` per entry: keep | archive | mark_bad | edge_case | skip. "
                "Optionally add `notes`. Save as verdicts.json (drop the `.template`). "
                "Then run: python -m openpi.scripts.tag_from_review verdicts.json"
            ),
            "queue_dir": str(out_dir),
            "verdicts": verdict_entries,
        },
        indent=2,
    ))
    logger.info("wrote %d contact sheets + queue.json + verdicts.template.json to %s", len(queue_entries), out_dir)
    print(f"\nReview queue: {queue_path}")
    print(f"Contact sheets: {out_dir}/*.png  ({len(queue_entries)} files)")
    print(f"Verdicts template: {verdicts_template_path}")
    print(f"  → copy to verdicts.json, fill in actions, then run tag_from_review.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
