
INPUT_FILE=/mnt/green-efs/kostiantyn.omelianchuk/gec_sota/data/evalsets/bea-dev.txt
M2_FILE=/mnt/green-efs/kostiantyn.omelianchuk/gec_sota/data/evalsets/bea-dev.m2  
#PRED_FILE=/mnt/green-efs/kostiantyn.omelianchuk/gec_sota/data/baselines_preds/YOUR_PRED_FILE.txt
PRED_FILE=/home/oleksandr.korniienko/errant_eval/llama-2-7b-chat-hf-batch-4-acc-2-lr-1e-05-up-800-wup-160-seed-801-gec_sota-clang60-others40/wi_locness.dev.gold.bea18.pred.txt

docker run -it --rm \
    -v $INPUT_FILE:/data/input.txt \
    -v $M2_FILE:/data/ref.m2 \
    -v $PRED_FILE:/data/pred.txt \
    errant

