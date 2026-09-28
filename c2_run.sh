#!/usr/bin/env bash
# c2 driver: feed -> CLIP ground truth -> renderer sweep. ONE JSON verdict on stdout (from c2_sweep.py).
set -euo pipefail
cd "$(dirname "$0")"

python3 c2_feed.py                                        # lavfi animation feed -> c2_frames/
~/venvs/elephant-gpu/bin/python c2_clip.py                # CLIP zero-shot labels (GPU if free)
python3 c2_sweep.py                                       # sweep + verdict -> stdout JSON
