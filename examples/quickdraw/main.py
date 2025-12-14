"""Client to query a QuickDraw policy server and render strokes on a canvas.

Key features
- Opens a websocket connection to the policy server.
- Cycles through a batch of predefined dummy prompts.
- Starts each prompt with a blank white canvas.
- Sends inputs matching `QuickDrawInputs` (sketch_image, state, prompt).
- Executes returned actions on the canvas (including lift-pen tokens).
- Optional debug mode saves every step; final canvas is always saved.
- Stops after a user-defined maximum number of strokes.
"""

from __future__ import annotations

import colorsys
import dataclasses
import pathlib
import time
from typing import Iterable

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image, ImageDraw
import tyro

# Lift pen token used by the QuickDraw policy/dataset
LIFT_PEN_TOKEN = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
_DEFAULT_STROKE_COLOR = (0, 0, 0)


@dataclasses.dataclass
class Args:
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000

    # Drawing control
    drawing_frequency: float = 10.0  # Hz; strokes per second
    max_strokes: int = 256  # Total strokes to execute before stopping
    debug: bool = False  # Save every step if True
    color_by_time: bool = False  # If True, color strokes by stroke index

    # Canvas / output
    image_size: int = 256
    output_dir: pathlib.Path = pathlib.Path("data/quickdraw/runs")


def _is_lift_pen(action: np.ndarray) -> bool:
    """Check if an action is the lift-pen token."""
    action = np.asarray(action).flatten()
    if action.shape[0] < 2:
        return False
    return bool(np.any(action[:2] < 0))


def _make_blank_canvas(image_size: int) -> np.ndarray:
    """Return a white canvas (H, W, 3) uint8."""
    return np.ones((image_size, image_size, 3), dtype=np.uint8) * 255


def _draw_step(
    canvas: np.ndarray,
    prev_pos: np.ndarray,
    action: np.ndarray,
    pen_down: bool,
    stroke_color: tuple[int, int, int] = _DEFAULT_STROKE_COLOR,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply a single action to the canvas.

    Args:
        canvas: Current canvas (H, W, 3) uint8.
        prev_pos: Previous (x, y) position.
        action: [x, y, pen_down] or lift-pen token.
        pen_down: Whether pen was down before this action.
        stroke_color: RGB color used when drawing this stroke.

    Returns:
        updated_canvas, new_pos, new_pen_down
    """
    action = np.asarray(action, dtype=np.float32).flatten()

    if _is_lift_pen(action):
        return canvas, prev_pos, False

    x, y, pd = action[:3]
    new_pos = np.array([x, y], dtype=np.float32)
    new_pen_down = pd > 0.5

    # Clamp to valid pixel range
    new_pos = np.clip(new_pos, 0, canvas.shape[0] - 1)

    # Draw if pen stays down across the step
    if pen_down and new_pen_down:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.line(
            [(float(prev_pos[0]), float(prev_pos[1])), (float(new_pos[0]), float(new_pos[1]))],
            fill=stroke_color,
            width=2,
        )
        canvas = np.array(img, dtype=np.uint8)

    return canvas, new_pos, new_pen_down


def _dummy_prompts() -> Iterable[str]:
    """Return a small batch of prompts to cycle through."""
    return [
        "Draw me a cat",
        "Sketch a house",
        "Create a drawing of a car",
        "Please draw a tree",
        "Show me how to draw a dog",
    ]


def _save_canvas(canvas: np.ndarray, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def _stroke_color_for_index(stroke_index: int, total_strokes: int) -> tuple[int, int, int]:
    """Return a time-based stroke color scaled to total strokes."""
    if total_strokes <= 1:
        return _DEFAULT_STROKE_COLOR
    idx = min(max(stroke_index, 0), total_strokes - 1)
    t = idx / float(total_strokes - 1)
    # Sweep hue from blue (2/3) to red (0); endpoints stay consistent across totals.
    hue = (1.0 - t) * (2.0 / 3.0)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(255 * r), int(255 * g), int(255 * b))


def run(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    hz = max(args.drawing_frequency, 1e-3)
    step_period = 1.0 / hz

    prompts = list(_dummy_prompts())

    for prompt_idx, prompt in enumerate(prompts):
        canvas = _make_blank_canvas(args.image_size)
        pen_down = False
        pos = np.array([0.0, 0.0], dtype=np.float32)
        strokes_done = 0
        actions_from_chunk_completed = 0
        action_chunk: np.ndarray | None = None

        # Directory per prompt
        run_dir = args.output_dir / f"prompt_{prompt_idx:02d}"
        debug_dir = run_dir / "steps"

        while strokes_done < args.max_strokes:
            # Refill action chunk when exhausted
            if action_chunk is None or actions_from_chunk_completed >= len(action_chunk):
                request = {
                    "sketch_image": canvas,
                    "state": np.array([pos[0], pos[1], float(pen_down)], dtype=np.float32),
                    "prompt": prompt,
                }
                action_chunk = np.asarray(policy.infer(request)["actions"])
                actions_from_chunk_completed = 0

            action = action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            stroke_color = (
                _stroke_color_for_index(strokes_done, args.max_strokes)
                if args.color_by_time
                else _DEFAULT_STROKE_COLOR
            )
            strokes_done += 1

            canvas, pos, pen_down = _draw_step(canvas, pos, action, pen_down, stroke_color)

            # Save step if debug
            if args.debug:
                _save_canvas(canvas, debug_dir / f"step_{strokes_done:04d}.png")

            # Respect drawing frequency
            time.sleep(step_period)

            if strokes_done >= args.max_strokes:
                break

        # Always save final canvas
        _save_canvas(canvas, run_dir / "final.png")
        print(f"Saved prompt '{prompt}' drawing to {run_dir / 'final.png'}")


if __name__ == "__main__":
    run(tyro.cli(Args))

