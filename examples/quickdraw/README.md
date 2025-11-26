# QuickDraw Dataset

Converts Google QuickDraw simplified ndjson sketches to LeRobot format.

## Model Input/Output

**Input:**
- `sketch_image`: RGB image `(256, 256, 3)` - rasterized drawing
- `prompt`: Natural language instruction (e.g., "Draw me a cat")

**Output:**
- `actions`: `(max_points, 3)` array of `[x, y, pen_down]` where `x,y` in [0,255], `pen_down` is 1.0 (draw) or 0.0 (lift)
- `point_mask`: `(max_points,)` mask for valid points
- Lift pen token `[-1.0, -1.0, 0.0]` inserted between strokes

## Converting quickdraw dataset to LeRobot data

**Convert dataset:**
```bash
uv run examples/quickdraw/convert_quickdraw_data_to_lerobot.py \
    --data_dir /path/to/quickdraw/ndjson/files \
    --max_drawings 1000
```

**Debug mode:**
```bash
uv run examples/quickdraw/convert_quickdraw_data_to_lerobot.py \
    --data_dir /path/to/quickdraw/ndjson/files \
    --debug
```

Dataset saved to `$HF_LEROBOT_HOME/quickdraw/`

