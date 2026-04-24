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

INPUT_JSON="../extracted_concepts_improved_full.json"
CONCEPTNET_CSV="/scratch/qyang129/conceptnet/conceptnet-assertions-5.7.0.csv"
SQLITE_DB="conceptnet_en_neighbors.sqlite"
OUTPUT_DIR="kg_snowflake"

# -----------------------------
# Chunking
# -----------------------------
CHUNK=3000

START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

echo "===== Job info ====="
echo "JobID:        ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Range:        ${START}-${END}"
echo "===================="

python build_subgraph.py \
  --input_json "${INPUT_JSON}" \
  --conceptnet_csv "${CONCEPTNET_CSV}" \
  --sqlite_db "${SQLITE_DB}" \
  --output_dir "${OUTPUT_DIR}" \
  --start "${START}" \
  --end "${END}"

echo "===== Done ====="