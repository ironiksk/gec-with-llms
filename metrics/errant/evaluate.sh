
python3 /errant/parallel_to_m2.py -orig /data/input.txt -cor /data/pred.txt -out /data/pred.m2
python3 /errant/compare_m2.py -hyp /data/pred.m2 -ref /data/ref.m2
