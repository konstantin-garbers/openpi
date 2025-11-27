import dataclasses
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Lift pen token: [-1.0, -1.0, 0.0] indicates pen should be lifted
LIFT_PEN_TOKEN = np.array([-1.0, -1.0, 0.0])


def is_lift_pen_token(action: np.ndarray) -> bool:
    """Check if an action is the lift pen token."""
    return action[0] < 0 or action[1] < 0


def make_quickdraw_example() -> dict[str, Any]:
    """Creates a random input example for the QuickDraw policy.
    
    Returns an example of the input format expected by QuickDrawInputs.
    
    Returns:
        Example input dict with keys:
            - "sketch_image": (256, 256, 3) uint8 array
            - "state": (3,) float32 array [x, y, pen_down]
            - "prompt": str natural language instruction
    """
    return {
        "sketch_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "state": np.array([128.0, 128.0, 1.0]),  # Current cursor position and pen state [x, y, pen_down]
        "prompt": "Draw me a cat",
    }


def _parse_image(image: np.ndarray | Any) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format.
    
    Converts images from various formats to the standard format expected by the model:
    - Handles float32 images in [0, 1] range → converts to uint8 [0, 255]
    - Handles (C, H, W) format → converts to (H, W, C)
    - Preserves (H, W, C) uint8 format
    
    Args:
        image: Input image in any of:
            - (H, W, C) uint8 array
            - (C, H, W) float32 array in [0, 1]
            - (C, H, W) uint8 array
    
    Returns:
        (H, W, C) uint8 array with values in [0, 255]
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class QuickDrawInputs(transforms.DataTransformFn):
    """
    Inputs for the QuickDraw policy.

    Converts QuickDraw dataset/robot format to model input format.

    **Input Format (from robot/dataset):**
    ```python
    {
        "sketch_image": np.ndarray,  # (H, W, 3) uint8 or (3, H, W) float32
                                     # Current drawing/canvas image
        "state": np.ndarray,         # Optional: (3,) float32 [x, y, pen_down]
                                     # Current cursor position and pen state
                                     # If not provided, extracted from actions
        "actions": np.ndarray,        # Optional (training only): 
                                     # (max_points, 3) or (action_horizon, 3) float32
                                     # [x, y, pen_down] actions
        "prompt": str,               # Optional: Natural language instruction
        "task": str,                 # Optional: Alternative key for prompt
    }
    ```

    **Output Format (to model):**
    ```python
    {
        "state": np.ndarray,         # (3,) float32 [x, y, pen_down]
                                     # Current cursor position and pen state
        "image": {
            "base_0_rgb": np.ndarray,      # (H, W, 3) uint8 - sketch image
            "left_wrist_0_rgb": np.ndarray, # (H, W, 3) uint8 - zeros (padding)
            "right_wrist_0_rgb": np.ndarray, # (H, W, 3) uint8 - zeros (padding)
        },
        "image_mask": {
            "base_0_rgb": bool,            # True - valid image
            "left_wrist_0_rgb": bool,       # True for PI0_FAST, False for PI0
            "right_wrist_0_rgb": bool,      # True for PI0_FAST, False for PI0
        },
        "actions": np.ndarray,       # Optional (training only):
                                     # (max_points, 3) or (action_horizon, 3) float32
        "prompt": str,               # Natural language instruction
    }
    ```
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Parse sketch_image to uint8 (H, W, C) format
        # LeRobot stores images as float32 (C, H, W), so we need to convert
        base_image = _parse_image(data["sketch_image"])

        zero_image = np.zeros_like(base_image)

        # Extract current cursor position and pen state from actions for state
        # State is [x, y, pen_down] representing the current cursor position and pen state
        # During training: extract from first valid action (or use origin with pen up if starting)
        # During inference: state should be provided by environment
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Find first valid (non-lift-pen) action to get current position and pen state
            # If no valid action found, default to origin with pen up (0, 0, 0)
            current_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            # Look for first valid action (not a lift pen token)
            for action in actions:
                if not is_lift_pen_token(action):
                    # Extract [x, y, pen_down] from first valid action
                    current_state = action[:3].astype(np.float32)
                    break
        else:
            # During inference, state should be provided
            # If not provided, default to origin with pen up
            if "state" in data:
                state_data = np.asarray(data["state"])
                if state_data.size >= 3:
                    current_state = state_data[:3].astype(np.float32)
                elif state_data.size >= 2:
                    # If only [x, y] provided, assume pen is up (0.0)
                    current_state = np.concatenate([state_data[:2].astype(np.float32), np.array([0.0])])
                else:
                    current_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                current_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        inputs = {
            # State contains current cursor position and pen state [x, y, pen_down]
            "state": current_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zero_image,
                "right_wrist_0_rgb": zero_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask padding images for pi0-FAST, but not for pi0
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training
        # QuickDraw actions are (max_points, 3) = (512, 3) with [x, y, pen_down]
        # The model expects (action_horizon, action_dim)
        # The data loader will handle slicing to action_horizon from the sequence
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Actions shape: (max_points, 3) = (512, 3) or (action_horizon, 3)
            # The data loader slices sequences to action_horizon automatically
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class QuickDrawOutputs(transforms.DataTransformFn):
    """
    Outputs for the QuickDraw policy.

    Converts model outputs back to robot/dataset format.

    **Input Format (from model):**
    ```python
    {
        "state": np.ndarray,         # (3,) float32 [x, y, pen_down]
                                     # Current state (passed through)
        "actions": np.ndarray,        # (action_horizon, action_dim) float32
                                     # Model predictions, e.g., (16, 7)
                                     # We extract first 3 dims: [x, y, pen_down]
        "point_mask": np.ndarray,    # Optional: (action_horizon,) int32 or bool
                                     # Mask for valid actions (training only)
    }
    ```

    **Output Format (to robot):**
    ```python
    {
        "actions": np.ndarray,        # (action_horizon, 3) or (N,) float32
                                     # [x, y, pen_down] actions
                                     # If point_mask provided, filtered to valid actions
                                     # Actions are absolute positions [0-255, 0-255, 0|1]
                                     # Lift pen token: [-1.0, -1.0, 0.0]
    }
    ```

    Notes:
    - Extracts first 3 action dimensions (x, y, pen_down)
    - Outputs absolute actions (as stored in dataset)
    - Handles lift pen tokens (preserved as [-1.0, -1.0, 0.0])
    - Applies point_mask to filter valid actions if available (training)
    """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"][:, :3])

        # Apply point_mask if available to filter valid actions
        # Note: During inference, point_mask may not be available
        # The mask is mainly used during training to filter padding
        if "point_mask" in data:
            mask = np.asarray(data["point_mask"]).astype(bool)
            # Only return actions where mask is True
            # Note: This assumes mask length matches action_horizon
            if len(mask) == len(actions):
                actions = actions[mask]

        return {"actions": actions}

