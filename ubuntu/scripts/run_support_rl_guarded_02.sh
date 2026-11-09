#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
trap 'status=$?; echo "$status" > logs/support_rl_guarded_02.exit' EXIT
store=/data/humanoid-sitting-perception
renderer='--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false'
bash scripts/run.sh python ../scripts/train_support_rl_guarded.py --mode train --iterations 96 \
  --output "$store/support_rl_diverse_train_02" --num-envs 16 --geometry-count 16 \
  --geometry-seed 51317 --seed 51319 --device cuda:0 --kit_args="$renderer" \
  > logs/support_rl_diverse_train_02.log 2>&1
evaluate_pair() {
  local bank="$1" envs="$2" count="$3" geometry_seed="$4" start_seed="$5"
  local extra=()
  if test "$bank" = balanced; then
    extra=(--bank-from "$store/support_rl_diverse_train_02/geometry_bank.json")
  fi
  for mode in zero evaluate; do
    local name="support_rl_${mode}_${bank}_02"
    local checkpoint=()
    if test "$mode" = evaluate; then
      checkpoint=(--checkpoint "$store/support_rl_diverse_train_02/final.pt")
    fi
    bash scripts/run.sh python ../scripts/train_support_rl_guarded.py --mode "$mode" \
      --output "$store/$name" --num-envs "$envs" --geometry-count "$count" \
      --geometry-seed "$geometry_seed" --seed "$start_seed" --device cuda:0 \
      --kit_args="$renderer" "${extra[@]}" "${checkpoint[@]}" > "logs/$name.log" 2>&1
    test -e "runs/$name" || ln -s "$store/$name" "runs/$name"
  done
}
evaluate_pair regression 8 4 51103 51107
evaluate_pair fresh 16 8 51403 51407
evaluate_pair balanced 16 16 51317 51319
test -e runs/support_rl_diverse_train_02 || ln -s "$store/support_rl_diverse_train_02" runs/support_rl_diverse_train_02
bash scripts/run.sh python ../scripts/report_support_rl_suite.py \
  --output "$store/support_rl_suite_results_02" > logs/support_rl_suite_results_02.log 2>&1
test -e runs/support_rl_suite_results_02 || ln -s "$store/support_rl_suite_results_02" runs/support_rl_suite_results_02
bash scripts/run.sh python ../scripts/audit_shared_frame_integrity.py \
  --output "$store/support_rl_suite_integrity_02.json" > logs/support_rl_suite_integrity_02.log 2>&1
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of json \
  "$store/support_rl_suite_results_02/sitting_comparison.mp4" > "$store/support_rl_suite_results_02/video_metadata.json"
ffmpeg -v error -i "$store/support_rl_suite_results_02/sitting_comparison.mp4" \
  -f null - > logs/support_rl_suite_video_decode_02.log 2>&1
