#!/bin/bash

FILES="Above_A Above_C Above_E Above_F Above_G Below_A Below_C Below_E Below_F Below_G"
MODELS="chatgpt,claude"
OUT_DIR="outputs"

GUIDES=(
  "data/guides/1_first_species/Fux Rules- 1vs1 note - Schubert v1.7.txt"
  "data/guides/2_second_species/Cantus 2notes vs 1 - Schubert version 2.0.txt"
  "data/guides/3_third_species/Fux Rules- 3rd species - 4vs1 - Schubert v2.0.txt"
  "data/guides/4_fourth_species/Fux Rules- 4th species - Syncopatio - Schubert v1.7.txt"
  "data/guides/5_fifth_species/Fux Rules- 5th species - Florish - Schubert v2.4.txt"
)

for GUIDE in "${GUIDES[@]}"; do
  echo "========== Starting batch for $GUIDE =========="
  poetry run python src/llm_fux/cli/run_batch.py \
    --models "$MODELS" \
    --context \
    --guide "$GUIDE" \
    --datatypes musicxml \
    --files $FILES \
    --use-batch-api \
    --outputs-dir "$OUT_DIR"
done
