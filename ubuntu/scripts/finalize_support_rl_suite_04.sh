#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_suite_finalize_04.exit' EXIT
store=/data/humanoid-sitting-perception
suite="$store/support_rl_suite_results_04"
bash scripts/run.sh python ../scripts/report_support_rl_suite.py --output "$suite" \
  > logs/support_rl_suite_results_04.log 2>&1
test -e runs/support_rl_suite_results_04 || ln -s "$suite" runs/support_rl_suite_results_04
cp scripts/report_support_rl_suite.py "$suite/report_source.py"
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of json \
  "$suite/sitting_comparison.mp4" > "$suite/video_metadata.json"
ffmpeg -v error -i "$suite/sitting_comparison.mp4" -f null - > logs/support_rl_suite_video_decode_04.log 2>&1
bash scripts/run.sh python ../scripts/audit_support_rl_suite.py --suite "$suite" \
  > logs/support_rl_suite_bundle_audit_04.log 2>&1
