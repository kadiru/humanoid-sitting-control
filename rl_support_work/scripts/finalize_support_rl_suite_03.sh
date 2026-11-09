#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_suite_finalize_03.exit' EXIT
store=/data/humanoid-sitting-perception
suite="$store/support_rl_suite_results_03"
bash scripts/run.sh python ../scripts/report_support_rl_suite.py --output "$suite" \
  > logs/support_rl_suite_results_03.log 2>&1
test -e runs/support_rl_suite_results_03 || ln -s "$suite" runs/support_rl_suite_results_03
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py \
  --output "$store/support_rl_suite_integrity_02.json" > logs/support_rl_suite_integrity_02.log 2>&1
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of json \
  "$suite/sitting_comparison.mp4" > "$suite/video_metadata.json"
ffmpeg -v error -i "$suite/sitting_comparison.mp4" -f null - > logs/support_rl_suite_video_decode_03.log 2>&1
bash scripts/run.sh python ../scripts/audit_support_rl_suite.py --suite "$suite" \
  > logs/support_rl_suite_bundle_audit_03.log 2>&1
