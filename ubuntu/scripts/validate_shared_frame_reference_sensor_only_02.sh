#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'echo $? > logs/shared_frame_reference_sensor_only_02.exit' EXIT
storage=/data/humanoid-sitting-perception
bash scripts/run.sh python ../scripts/run_shared_frame.py --capture "$storage/shared_frame_integrity_sensor_fixture_02" --output "$storage/shared_frame_reference_sensor_only_02" --max-frames 90
ln -s "$storage/shared_frame_reference_sensor_only_02" runs/shared_frame_reference_sensor_only_02
