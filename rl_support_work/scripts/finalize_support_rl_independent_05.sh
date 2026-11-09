#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_independent_finalize_05.exit' EXIT
store=/data/humanoid-sitting-perception
out="$store/support_rl_independent_results_05"
bash scripts/run.sh python ../scripts/report_support_rl_independent.py --output "$out" \
  > logs/support_rl_independent_results_05.log 2>&1
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of json \
  "$out/sitting_comparison.mp4" > "$out/video_metadata.json"
ffmpeg -v error -i "$out/sitting_comparison.mp4" -f null - > logs/support_rl_independent_video_decode_05.log 2>&1
cp docs/SUPPORT_RL_INDEPENDENT_PROTOCOL.md "$out/"
cp logs/support_rl_sampling_tests_04.log "$out/"
cp "$store/support_rl_independent_integrity_04.json" "$out/"
cp -r "$store/support_rl_startup_profile_01" "$out/startup_profile"
test -e runs/support_rl_independent_results_05 || ln -s "$out" runs/support_rl_independent_results_05
