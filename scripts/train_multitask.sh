#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config configs/default.yaml
