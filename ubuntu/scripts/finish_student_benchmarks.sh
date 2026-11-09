#!/usr/bin/env bash
set -euo pipefail
cd /home/kadir/humanoid-sitting
for spec in 'teacher fresh 24001' 'student fresh 24001' 'student blind 10514' 'student no_clock 10514' 'student no_balance 10514'; do
 read -r controller mode bank <<< "$spec"
 flags=(--geometry-stage 2 --geometry-seed "$bank" --geometry-count 16 --compact-recording --summary-only --device cuda:0)
 if [[ "$bank" == 24001 ]]; then flags+=(--seeds 3201 3202 3203); else flags+=(--seeds 3107 3108 3109); fi
 if [[ "$controller" == teacher ]]; then name=student_fresh_teacher_01; else
  flags+=(--student-checkpoint ../runs/student_dagger_01/best.pt)
  name=student_dagger_${mode}_01
  if [[ "$mode" != fresh ]]; then
   ablation="$mode"; [[ "$mode" == blind ]] && ablation=blind_geometry
   flags+=(--student-ablation "$ablation")
  fi
 fi
 bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py "${flags[@]}" --output "../runs/$name" > "runs/learned_sitting_progress_01/$name.log" 2>&1
 done
bash scripts/run.sh python ../scripts/evaluate_geometry_reference.py --checkpoint ../baselines/geometry-conditioned-rigid-v1/checkpoints/preserved_policy_129.pt --remove-teacher-reference --geometry-stage 2 --geometry-seed 10514 --geometry-count 16 --seeds 3107 3108 3109 --compact-recording --summary-only --output ../runs/old_ppo_no_reference_01 --device cuda:0 > runs/learned_sitting_progress_01/old_ppo_no_reference.log 2>&1
