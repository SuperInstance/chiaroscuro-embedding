#!/usr/bin/env bash
# c3 driver: 20fps master feed -> CLIP ground truth (GPU, serialized) -> sweep. ONE JSON verdict on stdout (from c3_sweep.py).
set -euo pipefail
cd "$(dirname "$0")"

python3 c3_feed.py                                        # 20fps master timeline -> c3_frames/
~/venvs/elephant-gpu/bin/python c3_clip.py                # CLIP zero-shot labels (fp16 GPU if free)
python3 c3_sweep.py                                       # 12-cell sweep + verdict -> stdout JSON
