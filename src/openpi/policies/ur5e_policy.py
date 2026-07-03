# OpenPI policy transforms for the Relling UR5e dataset.

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import ur5e_schema as _ur5e_schema


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["joints"])
        velocity_raw = data.get("joints_velocity")
        velocity = np.asarray(velocity_raw) if velocity_raw is not None else np.zeros_like(joints)
        state = _ur5e_schema.assemble_state(joints, velocity, data["gripper"])

        base_image = _parse_image(data[_ur5e_schema.CAMERA_FIELDS[0]])
        wrist_image = _parse_image(data[_ur5e_schema.CAMERA_FIELDS[1]])
        overhead_raw = data.get(_ur5e_schema.CAMERA_FIELDS[2])
        overhead_image = _parse_image(overhead_raw) if overhead_raw is not None else None
        overhead_present = overhead_image is not None and bool(np.any(np.asarray(overhead_image)))

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (
                    base_image,
                    wrist_image,
                    overhead_image if overhead_present else np.zeros_like(base_image),
                )
                image_masks = (
                    np.True_,
                    np.True_,
                    np.True_ if overhead_present else np.False_,
                )
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (
                    base_image,
                    overhead_image if overhead_present else np.zeros_like(base_image),
                    wrist_image,
                )
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
