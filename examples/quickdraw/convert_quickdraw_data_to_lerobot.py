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


# Sentinel value for "lift pen" token - coordinates outside valid range (0-255)
# Format: [x, y, t, pen_down] but we only use x, y, pen_down here (t is added in flatten_points)
LIFT_PEN_TOKEN = (-1.0, -1.0, 0.0)


def is_lift_pen_token(action: np.ndarray) -> bool:
    """Check if an action is the lift pen token."""
    return action[0] < 0 or action[1] < 0


def flatten_points(drawing: list[list[list[float]]], max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten strokes to a fixed-length sequence of [x, y, pen_down] points.
    
    Works with simplified QuickDraw data (no timing info, pre-normalized to 0-255).
    Normal drawing actions: [x, y, 1.0] where:
        - x, y are coordinates (0-255)
        - pen_down=1.0 indicates pen is down (drawing)
    
    After each stroke, appends a special "lift pen" token: [-1.0, -1.0, 0.0].
    The lift pen token indicates the pen should be lifted before the next action.
    
    """
    flattened = []
    
    for stroke_idx, stroke in enumerate(drawing):
        if len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if len(xs) != len(ys):
            raise ValueError("Stroke x/y arrays must have the same length.")
        
        # Add all points in the stroke with pen_down=1 (drawing actions)
        for x, y in zip(xs, ys):
            flattened.append((x, y, 1.0))  # [x, y, pen_down=1]
        
        # After each stroke (except possibly the last), append lift pen token
        # This tells the model to lift the pen before the next stroke
        if stroke_idx < len(drawing) - 1:  # Don't add lift token after last stroke
            flattened.append(LIFT_PEN_TOKEN)

    padded_points = np.zeros((max_points, 3), dtype=np.float32)  # [x, y, pen_down]
    mask = np.zeros((max_points,), dtype=np.int32)
    if not flattened:
        return padded_points, mask

    arr = np.asarray(flattened, dtype=np.float32)
    arr = arr[:max_points]

    padded_points[: len(arr)] = arr
    mask[: len(arr)] = 1
    return padded_points, mask


def render_drawing(drawing: list[list[list[float]]], image_size: int) -> np.ndarray:
    """Rasterize strokes to a grayscale image.
    
    Simplified dataset coordinates are already in 0-255 range, so we scale
    them to the target image_size.
    """
    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    # Simplified data is already 0-255, scale to target image size
    scale = image_size / 256.0

    for stroke in drawing:
        if len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if len(xs) < 2:
            continue
        scaled_points = [(float(x) * scale, float(y) * scale) for x, y in zip(xs, ys)]
        draw.line(scaled_points, fill=0, width=3)

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
    image: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    word: str,
    output_dir: Path,
    index: int,
) -> None:
    """Save debugging visualizations of a drawing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the rasterized image
    img_path = output_dir / f"{index:04d}_{word}_image.png"
    Image.fromarray(image).save(img_path)

    # Save a visualization of the stroke sequence
    vis_image = Image.new("RGB", (256, 256), color="white")
    vis_draw = ImageDraw.Draw(vis_image)

    valid_points = actions[mask.astype(bool)]
    if len(valid_points) > 0:
        xs = valid_points[:, 0]
        ys = valid_points[:, 1]

        # Lift-pen tokens have negative coords; keep them so we can break segments
        valid_mask = (xs >= 0) & (ys >= 0)
        pen_down = valid_points[:, 2]

        for i in range(len(valid_points) - 1):
            # Skip drawing if either endpoint is a lift token
            if not (valid_mask[i] and valid_mask[i + 1]):
                continue
            if not (pen_down[i] > 0.5 and pen_down[i + 1] > 0.5):
                continue

            # Coordinates are already in 0-255 range, clamp to ensure valid pixel values
            x0 = int(np.clip(valid_points[i, 0], 0, 255))
            y0 = int(np.clip(valid_points[i, 1], 0, 255))
            x1 = int(np.clip(valid_points[i + 1, 0], 0, 255))
            y1 = int(np.clip(valid_points[i + 1, 1], 0, 255))

            # Color from blue (start) to red (end) based on position in sequence
            color_ratio = i / max(len(valid_points) - 1, 1)
            color = (
                int(255 * color_ratio),
                0,
                int(255 * (1 - color_ratio)),
            )
            vis_draw.line([(x0, y0), (x1, y1)], fill=color, width=2)
        
        # Mark lift pen tokens with a special symbol
        lift_indices = np.where(~valid_mask)[0]
        for idx in lift_indices:
            # Draw a small "X" to indicate lift pen token
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
        if len(valid_points) > 0:
            # Separate drawing actions from lift pen tokens
            drawing_actions = valid_points[(valid_points[:, 0] >= 0) & (valid_points[:, 1] >= 0)]
            lift_tokens = valid_points[(valid_points[:, 0] < 0) | (valid_points[:, 1] < 0)]
            
            if len(drawing_actions) > 0:
                f.write(f"Drawing actions: {len(drawing_actions)}\n")
                f.write(f"X range: [{drawing_actions[:, 0].min():.1f}, {drawing_actions[:, 0].max():.1f}]\n")
                f.write(f"Y range: [{drawing_actions[:, 1].min():.1f}, {drawing_actions[:, 1].max():.1f}]\n")
            if len(lift_tokens) > 0:
                f.write(f"Lift pen tokens: {len(lift_tokens)}\n")


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
                "names": ["point", "axis"],  # axis: [x, y, pen_down]
                # Normal actions: [x, y, 1.0] where x,y in [0, 255]
                # Lift pen token: [-1.0, -1.0, 0.0]
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
        image = render_drawing(drawing, image_size)
        actions, mask = flatten_points(drawing, max_points)

        # Debug mode: save visualizations
        if debug and debug_saved < debug_max_samples:
            debug_save_drawing(
                drawing, image, actions, mask, entry["word"], debug_path, converted
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
