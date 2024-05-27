
INPUT_FILE=bea-dev.txt
M2_FILE=bea-dev.m2  
#PRED_FILE=YOUR_PRED_FILE.txt

docker run -it --rm \
    -v $INPUT_FILE:/data/input.txt \
    -v $M2_FILE:/data/ref.m2 \
    -v $PRED_FILE:/data/pred.txt \
    errant

