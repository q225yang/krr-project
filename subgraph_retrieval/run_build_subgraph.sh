#!/bin/bash
#SBATCH --job-name=build_subgraph
#SBATCH --partition=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=/scratch/qyang129/logs/kg_%x_%A_%a.out
#SBATCH --error=/scratch/qyang129/logs/kg_%x_%A_%a.err

set -euo pipefail

cd /home/qyang129/krr-project/subgraph_retrieval

# This is the previous image-only file.
# It is used only to decide which idx values to process.
INPUT_JSON="../extracted_concepts_improved.json"

# New Qwen entity files.
IMG_ENTITIES_JSON="../img_entities_qwen.json"
TEXT_ENTITIES_JSON="../text_entities_qwen.json"

CONCEPTNET_CSV="/scratch/qyang129/conceptnet/conceptnet-assertions-5.7.0.csv"
SQLITE_DB="conceptnet_en_neighbors.sqlite"
OUTPUT_DIR="kg_snowflake_qwen_combined"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /scratch/qyang129/logs

# Around 2000 image-only samples, so 3000 is fine.
# If you later increase data size, use array jobs.
CHUNK=3000

START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

echo "===== Job info ====="
echo "JobID:              ${SLURM_JOB_ID}"
echo "Array Task:         ${SLURM_ARRAY_TASK_ID}"
echo "Node:               $(hostname)"
echo "Start time:         $(date)"
echo "Range by position:  ${START}-${END}"
echo "Input filter:       ${INPUT_JSON}"
echo "Image entities:     ${IMG_ENTITIES_JSON}"
echo "Text entities:      ${TEXT_ENTITIES_JSON}"
echo "Output dir:         ${OUTPUT_DIR}"
echo "===================="

python build_subgraph.py \
  --input_json "${INPUT_JSON}" \
  --img_entities_json "${IMG_ENTITIES_JSON}" \
  --text_entities_json "${TEXT_ENTITIES_JSON}" \
  --conceptnet_csv "${CONCEPTNET_CSV}" \
  --sqlite_db "${SQLITE_DB}" \
  --output_dir "${OUTPUT_DIR}" \
  --start "${START}" \
  --end "${END}" \
  --max_anchors 8 \
  --max_leaves 5 \
  --neighbor_limit 50 \
  --min_weight 0.5 \
  --include_states

echo "===== Done ====="
echo "End time: $(date)"