"""
Minimal example script for converting Google Quick, Draw! simplified ndjson sketches to
LeRobot format so they can be loaded by the OpenPi training stack.

This script works with the simplified QuickDraw dataset, which has:
- No timing information (timestamps are generated sequentially)
- Pre-processed coordinates (aligned to top-left, scaled to 0-255)
- Resampled strokes with 1 pixel spacing
- Simplified strokes using Ramer-Douglas-Peucker algorithm

Usage:
uv run examples/quickdraw/convert_quickdraw_data_to_lerobot.py \
    --data_dir /path/to/quickdraw/ndjson/files \
    --max_drawings 1000

Debug mode (saves first few drawings as images):
uv run examples/quickdraw/convert_quickdraw_data_to_lerobot.py \
    --data_dir /path/to/quickdraw/ndjson/files \
    --debug \
    --debug_output_dir ./debug_output

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm
import tyro

REPO_NAME = "quickdraw"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MAX_POINTS = 512
DEFAULT_IMAGE_WRITER_THREADS = 4
DEFAULT_IMAGE_WRITER_PROCESSES = 2

# Natural language prompt templates for converting words to drawing instructions
# These templates provide variety and align with VLM training on natural language
DRAWING_PROMPT_TEMPLATES = [
    "Draw me a {word}",
    "Sketch a {word}",
    "Create a drawing of a {word}",
    "Can you draw a {word}?",
    "I want you to draw a {word}",
    "Please draw a {word}",
    "Draw a {word} for me",
    "Make a sketch of a {word}",
    "I'd like to see a {word} drawn",
    "Show me how to draw a {word}",
    "Draw a {word}",
    "Sketch out a {word}",
    "Create a sketch of a {word}",
    "Can you sketch a {word}?",
    "I need a drawing of a {word}",
    "Draw a picture of a {word}",
    "Make a drawing of a {word}",
    "Illustrate a {word}",
    "Draw a simple {word}",
    "Create a simple drawing of a {word}",
    "Sketch a simple {word}",
    "Draw a quick {word}",
    "Make a quick sketch of a {word}",
    "I want a {word} drawing",
    "Draw a {word} please",
    "Could you sketch a {word}?",
    "Let's draw a {word}.",
    "Please illustrate a {word}.",
    "Give me a sketch of a {word}.",
    "Make a simple drawing of a {word}.",
    "Show me a sketch of a {word}.",
    "How would you draw a {word}?",
    "What does a {word} look like? Draw it.",
    "Picture a {word} and draw it.",
    "Draw the shape of a {word}.",
    "Try drawing a {word} for me.",
    "Draw your best {word}.",
    "Show your skills: draw a {word}.",
    "Let's see a {word} sketch.",
    "Depict a {word} in a sketch.",
    "Draw something that looks like a {word}.",
    "Give me your version of a {word}.",
    "Create a quick doodle of a {word}.",
    "Draw an example of a {word}.",
    "Sketch your interpretation of a {word}.",
    "Represent a {word} by drawing it.",
]


def create_drawing_prompt(word: str, seed: int | None = None) -> str:
    """Convert a word into a natural language drawing prompt.
    
    Randomly selects from a variety of prompt templates to add diversity
    to the training data. This helps the model generalize better and aligns
    with how vision-language models are trained on natural language.
    
    Args:
        word: The word/category to draw (e.g., "cat", "dog", "car")
        seed: Optional random seed for reproducibility. If None, uses random selection.
    
    Returns:
        A natural language prompt like "Draw me a cat" or "Sketch a dog"
    
    Examples:
        >>> create_drawing_prompt("cat")
        "Draw me a cat"
        >>> create_drawing_prompt("car", seed=42)
        "Sketch a car"
    """
    if seed is not None:
        # Use seed to deterministically select template based on word
        # This ensures same word always gets same template (useful for reproducibility)
        rng = random.Random(hash(word) ^ seed)
    else:
        rng = random.Random()
    
    template = rng.choice(DRAWING_PROMPT_TEMPLATES)
    return template.format(word=word.lower())


# Pen state flag: -1.0 means lifted, >0 means pen is down
PEN_LIFTED_VALUE = -1.0


def is_lift_pen_token(action: np.ndarray) -> bool:
    """Check if an action lifts the pen (pen_down flag < 0)."""
    return action[2] < 0


def build_actions(
    drawing: list[list[list[float]]],
    *,
    use_relative_actions: bool = True,
) -> list[tuple[float, float, float]]:
    """Convert strokes into a sequence of actions (un-padded).

    use_relative_actions:
        - True (default): actions are deltas from the previous position.
        - False: actions are absolute coordinates in [0, 255].
    """
    actions: list[tuple[float, float, float]] = []
    prev_x, prev_y = 0.0, 0.0

    for stroke_idx, stroke in enumerate(drawing):
        if len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if len(xs) != len(ys):
            raise ValueError("Stroke x/y arrays must have the same length.")

        start_x, start_y = float(xs[0]), float(ys[0])

        # Move to the start of the stroke with the pen lifted
        if use_relative_actions:
            actions.append((start_x - prev_x, start_y - prev_y, PEN_LIFTED_VALUE))
            actions.append((0.0, 0.0, 1.0))  # drop the pen at the stroke start
        else:
            actions.append((start_x, start_y, PEN_LIFTED_VALUE))
            actions.append((start_x, start_y, 1.0))
        prev_x, prev_y = start_x, start_y

        # Draw the stroke with the pen down
        for x, y in zip(xs[1:], ys[1:]):
            x_f, y_f = float(x), float(y)
            if use_relative_actions:
                dx, dy = x_f - prev_x, y_f - prev_y
                actions.append((dx, dy, 1.0))
            else:  # absolute
                actions.append((x_f, y_f, 1.0))
            prev_x, prev_y = x_f, y_f

    return actions


def flatten_points(
    drawing: list[list[list[float]]],
    max_points: int,
    use_relative_actions: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten strokes to a fixed-length sequence of [x, y, pen_state] points.
    
    Works with simplified QuickDraw data (no timing info, pre-normalized to 0-255).
    Supports relative or absolute action encodings. pen_state > 0 means the pen
    is down, pen_state < 0 means the pen is lifted.
    
    """
    flattened = build_actions(drawing, use_relative_actions=use_relative_actions)

    padded_points = np.zeros((max_points, 3), dtype=np.float32)  # [x, y, pen_down]
    mask = np.zeros((max_points,), dtype=np.int32)
    if not flattened:
        return padded_points, mask

    arr = np.asarray(flattened, dtype=np.float32)
    arr = arr[:max_points]

    padded_points[: len(arr)] = arr
    mask[: len(arr)] = 1
    return padded_points, mask


def drawing_to_image(
    drawing: list[list[list[float]]],
    image_size: int,
) -> np.ndarray:
    """Rasterize the raw stroke coordinates to a grayscale RGB image."""
    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)

    for stroke in drawing:
        if len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if len(xs) != len(ys):
            raise ValueError("Stroke x/y arrays must have the same length.")
        points = list(zip(xs, ys))
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            x0_c = float(np.clip(x0, 0, image_size - 1))
            y0_c = float(np.clip(y0, 0, image_size - 1))
            x1_c = float(np.clip(x1, 0, image_size - 1))
            y1_c = float(np.clip(y1, 0, image_size - 1))
            draw.line([(x0_c, y0_c), (x1_c, y1_c)], fill=0, width=3)

    array = np.array(image, dtype=np.uint8)
    return np.repeat(array[..., None], 3, axis=-1)


def iter_ndjson_lines(paths: Sequence[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)


def debug_save_drawing(
    drawing: list[list[list[float]]],
    actions: np.ndarray,
    mask: np.ndarray,
    word: str,
    output_dir: Path,
    index: int,
    use_relative_actions: bool,
) -> None:
    """Save debugging visualizations of a drawing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a visualization of the stroke sequence
    vis_image = Image.new("RGB", (256, 256), color="white")
    vis_draw = ImageDraw.Draw(vis_image)

    valid_points = actions[mask.astype(bool)]
    if len(valid_points) > 0:
        pen_down_arr = valid_points[:, 2]

        pos = np.array([0.0, 0.0], dtype=np.float32)
        pen_is_down = False

        for i in range(len(valid_points)):
            action = valid_points[i]

            if use_relative_actions:
                new_pos = pos + action[:2]
            else:
                new_pos = action[:2]

            new_pen_down = pen_down_arr[i] > 0.5

            if i > 0:
                prev_action = valid_points[i - 1]
                if (not is_lift_pen_token(prev_action)) and pen_is_down and new_pen_down:
                    color_ratio = (i - 1) / max(len(valid_points) - 1, 1)
                    color = (
                        int(255 * color_ratio),
                        0,
                        int(255 * (1 - color_ratio)),
                    )
                    x0 = int(np.clip(pos[0], 0, 255))
                    y0 = int(np.clip(pos[1], 0, 255))
                    x1 = int(np.clip(new_pos[0], 0, 255))
                    y1 = int(np.clip(new_pos[1], 0, 255))
                    vis_draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

            pos = new_pos
            pen_is_down = new_pen_down

        # Mark lift actions with a special symbol
        lift_indices = np.where(valid_points[:, 2] < 0)[0]
        for idx in lift_indices:
            x_pos, y_pos = 10 + (idx % 20) * 12, 10 + (idx // 20) * 12
            vis_draw.line([(x_pos - 3, y_pos - 3), (x_pos + 3, y_pos + 3)], fill=(255, 0, 0), width=2)
            vis_draw.line([(x_pos - 3, y_pos + 3), (x_pos + 3, y_pos - 3)], fill=(255, 0, 0), width=2)

    vis_path = output_dir / f"{index:04d}_{word}_strokes.png"
    vis_image.save(vis_path)

    # Save metadata
    meta_path = output_dir / f"{index:04d}_{word}_meta.txt"
    with meta_path.open("w") as f:
        f.write(f"Word: {word}\n")
        f.write(f"Number of valid points: {mask.sum()}\n")
        f.write(f"Total points (with padding): {len(actions)}\n")
        f.write(f"Number of strokes: {len(drawing)}\n")
        f.write(f"Actions are relative: {use_relative_actions}\n")
        if len(valid_points) > 0:
            # Separate drawing actions from lift actions
            drawing_actions = valid_points[valid_points[:, 2] >= 0]
            lift_tokens = valid_points[valid_points[:, 2] < 0]
            
            if len(drawing_actions) > 0:
                f.write(f"Drawing actions: {len(drawing_actions)}\n")
                f.write(f"X range: [{drawing_actions[:, 0].min():.1f}, {drawing_actions[:, 0].max():.1f}]\n")
                f.write(f"Y range: [{drawing_actions[:, 1].min():.1f}, {drawing_actions[:, 1].max():.1f}]\n")
            if len(lift_tokens) > 0:
                f.write(f"Lift actions: {len(lift_tokens)}\n")


def main(
    data_dir: str,
    *,
    repo_name: str = REPO_NAME,
    categories: Sequence[str] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_drawings: int | None = None,
    debug: bool = False,
    debug_output_dir: str | None = None,
    debug_max_samples: int = 10,
    use_relative_actions: bool = True,
):
    """Convert simplified QuickDraw ndjson files to LeRobot format.

    Works with the simplified QuickDraw dataset format (pre-processed coordinates
    in 0-255 range, no timing information).

    Args:
        data_dir: Directory containing .ndjson files
        repo_name: LeRobot dataset repository name
        categories: Optional list of categories to filter (case-insensitive)
        max_points: Maximum number of points per drawing (for padding)
        image_size: Size of output images (square)
        max_drawings: Maximum number of drawings to convert (None = all)
        debug: Enable debug mode (saves visualizations)
        debug_output_dir: Directory to save debug outputs (default: ./debug_output)
        debug_max_samples: Maximum number of samples to save in debug mode
        use_relative_actions: If True (default), store relative deltas; else absolute
    """
    # Clean up existing dataset output
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    if categories:
        category_filter = {c.lower() for c in categories}
    else:
        category_filter = None

    data_dir_path = Path(data_dir)
    ndjson_files = sorted(data_dir_path.glob("*.ndjson"))
    if not ndjson_files:
        raise FileNotFoundError(f"No .ndjson files found in {data_dir_path}")

    # Setup debug output directory
    if debug:
        if debug_output_dir is None:
            debug_output_dir = "./debug_output"
        debug_path = Path(debug_output_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled. Saving samples to: {debug_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="quickdraw",
        # The quickdraw dataset is independent of fps, so we use the default fps of 5
        fps=5,
        features={
            "sketch_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (max_points, 3),
                "names": ["point", "axis"],  # axis: [x, y, pen_state]
                # pen_state > 0: pen down, pen_state < 0: pen lifted
                # Relative actions (default): deltas; Absolute: coordinates in [0, 255]
            },
            "point_mask": {
                "dtype": "int32",
                "shape": (max_points,),
                "names": ["point"],
            },
        },
        image_writer_threads=DEFAULT_IMAGE_WRITER_THREADS,
        image_writer_processes=DEFAULT_IMAGE_WRITER_PROCESSES,
    )

    converted = 0
    debug_saved = 0
    iterator = iter_ndjson_lines(ndjson_files)
    for entry in tqdm(iterator, desc="Converting drawings"):
        if category_filter and entry["word"].lower() not in category_filter:
            continue

        drawing = entry["drawing"]
        actions, mask = flatten_points(drawing, max_points, use_relative_actions=use_relative_actions)
        image = drawing_to_image(drawing, image_size)

        # Debug mode: save visualizations
        if debug and debug_saved < debug_max_samples:
            debug_save_drawing(
                drawing,
                actions,
                mask,
                entry["word"],
                debug_path,
                converted,
                use_relative_actions,
            )
            debug_saved += 1

        # Convert word to natural language prompt for VLM training
        prompt = create_drawing_prompt(entry["word"])
        
        dataset.add_frame(
            {
                "sketch_image": image,
                "actions": actions,
                "point_mask": mask,
                "task": prompt,  # Store natural language prompt instead of just word
            }
        )
        dataset.save_episode()
        converted += 1

        if max_drawings and converted >= max_drawings:
            break

    print(f"Converted {converted} drawings into LeRobot format.")
    if debug:
        print(f"Saved {debug_saved} debug visualizations to {debug_path}")


if __name__ == "__main__":
    tyro.cli(main)