#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1
trap 'status=$?; echo "$status" > logs/support_rl_suite_finalize_01.exit' EXIT
while ! test -f logs/support_rl_balanced_eval_01.exit; do sleep 15; done
test "$(cat logs/support_rl_balanced_eval_01.exit)" = 0
bash scripts/run.sh python ../scripts/report_support_rl_suite.py \
  --output /data/humanoid-sitting-perception/support_rl_suite_results_01 \
  > logs/support_rl_suite_results_01.log 2>&1
test -e runs/support_rl_suite_results_01 || ln -s /data/humanoid-sitting-perception/support_rl_suite_results_01 runs/support_rl_suite_results_01
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py \
  --output /data/humanoid-sitting-perception/support_rl_suite_integrity_01.json \
  > logs/support_rl_suite_integrity_01.log 2>&1
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration \
  -of json /data/humanoid-sitting-perception/support_rl_suite_results_01/sitting_comparison.mp4 \
  > /data/humanoid-sitting-perception/support_rl_suite_results_01/video_metadata.json
ffmpeg -v error -i /data/humanoid-sitting-perception/support_rl_suite_results_01/sitting_comparison.mp4 \
  -f null - > logs/support_rl_suite_video_decode_01.log 2>&1
