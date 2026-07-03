"""Single source of truth for the avea/openpi UR5e dataset schema.

Vendored into openpi from robot-research so openpi can load its configs
without robot-research on PYTHONPATH. Keep this the local SSOT: if the
dataset contract changes, this module and the robot-research writer must
be updated in lockstep.

Layout:
    state  = concat(joints[6], joints_velocity[6], gripper[1])  -> 13d
    action = (delta_joints[6], gripper[1])                      -> 7d
    images = base_rgb + wrist_rgb + overhead_rgb                -> 3 channels
"""

from dataclasses import dataclass
import json
import os
import pathlib

import numpy as np

DATASET_SCHEMA_VERSION = "v0.2"


@dataclass(frozen=True)
class _Field:
    name: str
    dim: int


STATE_FIELDS: tuple[_Field, ...] = (
    _Field("joints", 6),
    _Field("joints_velocity", 6),
    _Field("gripper", 1),
)

ACTION_FIELDS: tuple[_Field, ...] = (
    _Field("delta_joints", 6),
    _Field("gripper", 1),
)

CAMERA_FIELDS: tuple[str, ...] = (
    "base_rgb",
    "wrist_rgb",
    "overhead_rgb",
)

EXPECTED_STATE_DIM: int = sum(f.dim for f in STATE_FIELDS)
EXPECTED_ACTION_DIM: int = sum(f.dim for f in ACTION_FIELDS)

# Parquet column names (what writer emits and what RepackTransform reads).
PARQUET_STATE_COLUMNS: tuple[str, ...] = tuple(f.name for f in STATE_FIELDS)
PARQUET_ACTIONS_COLUMN: str = "actions"
PARQUET_TASK_COLUMN: str = "task"

# Mapping consumed by openpi.transforms.RepackTransform.
# Keys: post-repack names that UR5Inputs reads.
# Values: parquet column names.
OPENPI_REPACK_MAP: dict[str, str] = {
    **{cam: cam for cam in CAMERA_FIELDS},
    **{f.name: f.name for f in STATE_FIELDS},
    "actions": PARQUET_ACTIONS_COLUMN,
    "prompt": PARQUET_TASK_COLUMN,
}

# Delta-action mask: True dims become deltas relative to current state,
# False dims pass through as absolute. Joints become deltas; gripper stays
# absolute (open/close commands don't compose).
DELTA_ACTION_MASK: tuple[bool, ...] = tuple([True] * ACTION_FIELDS[0].dim + [False] * ACTION_FIELDS[1].dim)


def assemble_state(joints, joints_velocity, gripper):
    """Build the 13d state vector in the canonical order.

    Accepts numpy arrays or anything array-like. Returns a 1-d float32 numpy
    array. Handles scalar gripper by promoting to 1-d.
    """
    joints = np.asarray(joints, dtype=np.float32).reshape(-1)
    joints_velocity = np.asarray(joints_velocity, dtype=np.float32).reshape(-1)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if joints.shape != (STATE_FIELDS[0].dim,):
        raise ValueError(f"joints must be {STATE_FIELDS[0].dim}-d, got {joints.shape}")
    if joints_velocity.shape != (STATE_FIELDS[1].dim,):
        raise ValueError(f"joints_velocity must be {STATE_FIELDS[1].dim}-d, got {joints_velocity.shape}")
    if gripper.shape != (STATE_FIELDS[2].dim,):
        raise ValueError(f"gripper must be {STATE_FIELDS[2].dim}-d, got {gripper.shape}")
    return np.concatenate([joints, joints_velocity, gripper])


def parquet_features(image_size: tuple[int, int]):
    """Return the LeRobotDataset features dict.

    Built from STATE_FIELDS + CAMERA_FIELDS so the writer and the consumers
    cannot disagree on column names or shapes.
    """
    height, width = image_size
    image_spec = {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
    }
    features: dict[str, dict] = dict.fromkeys(CAMERA_FIELDS, image_spec)
    for f in STATE_FIELDS:
        features[f.name] = {
            "dtype": "float32",
            "shape": (f.dim,),
            "names": [f.name if f.dim == 1 else f"{f.name}_dim"],
        }
    features[PARQUET_ACTIONS_COLUMN] = {
        "dtype": "float32",
        "shape": (EXPECTED_ACTION_DIM,),
        "names": ["action"],
    }
    return features


def dataset_meta() -> dict:
    """Manifest written next to the parquet to identify the schema version."""
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "state_columns": list(PARQUET_STATE_COLUMNS),
        "camera_columns": list(CAMERA_FIELDS),
        "actions_column": PARQUET_ACTIONS_COLUMN,
        "actions_dim": EXPECTED_ACTION_DIM,
        "state_dim": EXPECTED_STATE_DIM,
    }


def assert_dataset_schema_version(repo_id: str, root: str | None = None) -> None:
    """Verify the dataset at HF_LEROBOT_HOME/repo_id matches DATASET_SCHEMA_VERSION.

    If root is None, falls back to HF_LEROBOT_HOME env, then the default LeRobot
    cache dir. Missing meta file = treated as an old (pre-v0.2) dataset and rejected.
    """
    if root is None:
        root = os.environ.get("HF_LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
    meta_path = pathlib.Path(root) / repo_id / "ur5e_dataset_meta.json"
    if not meta_path.exists():
        raise AssertionError(
            f"dataset {repo_id} is missing ur5e_dataset_meta.json at {meta_path}; "
            f"rebuild with the current cli.py (schema {DATASET_SCHEMA_VERSION})"
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise AssertionError(
            f"dataset {repo_id} schema_version={meta.get('schema_version')!r} != "
            f"expected {DATASET_SCHEMA_VERSION!r}; rebuild via "
            f"`python -m training.lerobot_ur5e.cli build --overwrite ...`"
        )
