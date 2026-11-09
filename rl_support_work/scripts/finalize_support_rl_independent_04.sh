#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_independent_finalize_04.exit' EXIT
while tmux has-session -t support_independent_05 2>/dev/null; do sleep 20; done
test "$(cat logs/support_rl_independent_05.exit)" = 0
store=/data/humanoid-sitting-perception
out="$store/support_rl_independent_results_04"
bash scripts/run.sh python ../scripts/report_support_rl_independent.py --output "$out" \
  > logs/support_rl_independent_results_04.log 2>&1
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py \
  --output "$store/support_rl_independent_integrity_04.json" > logs/support_rl_independent_integrity_04.log 2>&1
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of json \
  "$out/sitting_comparison.mp4" > "$out/video_metadata.json"
ffmpeg -v error -i "$out/sitting_comparison.mp4" -f null - > logs/support_rl_independent_video_decode_04.log 2>&1
cp docs/SUPPORT_RL_INDEPENDENT_PROTOCOL.md "$out/"
cp logs/support_rl_sampling_tests_04.log "$out/"
test -e runs/support_rl_independent_results_04 || ln -s "$out" runs/support_rl_independent_results_04
