!/bin/bash
HORIZONS=(5 10 30 50 100)
CONTEXT_LENGTHS=(256 512 1024)
FUSION_METHODS=(max mean min)

for horizon in "${HORIZONS[@]}"; do
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        for fusion in "${FUSION_METHODS[@]}"; do
            echo "=========================================="
            echo "Running horizon=${horizon}, context_length=${context_length}, fusion=${fusion}"
            echo "=========================================="

            python bidirectional.py \
                --split_ratio 0.3 \
                --horizon "${horizon}" \
                --context_length "${context_length}" \
                --gpu 0 \
                --score_method mse \
                --agg_method l2 \
                --fusion_method "${fusion}" \
                --smooth_window 10 \
                --sliding_window_VUS 100 \
                --vus_version opt \
                --vus_thre 250
        done
    done
done
