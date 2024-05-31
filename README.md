# LTM
Code for LTM.

Distributed Training on 4 GPUs

``````
./distributed_train.sh 4 'your-data-path' --model ltm_mobilevit_s \
-b 64 --workers 8 --pin-mem \
--sched cosine --epochs 300 --min-lr 0.00002 --lr 0.0002 --warmup-lr 0.0002 --warmup-epochs 5 \
--opt adamw --weight-decay 0.01 \
--model-ema --model-ema-decay 0.9995 \
--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp \
--smoothing 0.1 \
--checkpoint-hist 20 \
--compression-ratio 0.8 \
--output 'your-output-path-saving-results' \
``````




