#!/bin/bash
# =============================================================================
# Case Study: Fusion Weight Analysis
# Run case_study.py for each dataset in a separate tmux window
# =============================================================================

SESSION_NAME="case_study"
WORK_DIR="$HOME/AmazonReviews2023"

MODEL_NAME="Llama-3.2-1B-Instruct"
COMBO="itemknn_lightgcn_pop"
CHECKPOINT="${CHECKPOINT:-grpo}"
PRED_DIR="${PRED_DIR:-results}"
CEM_DIR="${CEM_DIR:-results/cem}"
PG_DIR="${PG_DIR:-results/pg}"
OUTPUT_DIR="${OUTPUT_DIR:-results/case_study}"
DATA_DIR="${DATA_DIR:-GRPO/data/pure_models}"
PROFILE_CUTOFF="${PROFILE_CUTOFF:-500000}"
DATA_PATH="${DATA_PATH:-dataset}"

DATASETS=("ml-1m" "steam" "Food")

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    CMD="cd ${WORK_DIR} && export PYTHONPATH=${WORK_DIR} && \
echo '========== Case Study: ${DATASET} ==========' && \
conda run -n grid python GRPO/analysis/case_study.py \
    --dataset ${DATASET} \
    --recaller_combo ${COMBO} \
    --model_name ${MODEL_NAME} \
    --checkpoint ${CHECKPOINT} \
    --pred_dir ${PRED_DIR} \
    --cem_dir ${CEM_DIR} \
    --pg_dir ${PG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --profile_cutoff ${PROFILE_CUTOFF} \
    --data_path ${DATA_PATH} && \
echo 'Done: ${DATASET}'; exec bash"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "${SESSION_NAME}" -n "${DATASET}" "${CMD}"
    else
        tmux new-window -t "${SESSION_NAME}" -n "${DATASET}" "${CMD}"
    fi
done

echo "Launched tmux session: ${SESSION_NAME}"
echo "  Windows: ${DATASETS[*]}"
echo "  Model:   ${MODEL_NAME}"
echo "  Combo:   ${COMBO}"
echo ""
echo "Attach with:  tmux attach -t ${SESSION_NAME}"
