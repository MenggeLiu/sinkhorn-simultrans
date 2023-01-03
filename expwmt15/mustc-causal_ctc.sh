#!/usr/bin/env bash

SRC=en
TGT=zh
DATA=~/simultaneous_translation/simul_confi/data_bins/mustc_${SRC}${TGT}
WANDB_START_METHOD=thread
FAIRSEQ=~/simultaneous_translation/sinkhorn-simultrans/fairseq
USERDIR=~/simultaneous_translation/sinkhorn-simultrans/simultaneous_translation
PYTHONPATH="$FAIRSEQ:$PYTHONPATH"


DELAY=1
TASK=ctc_delay${DELAY}

save_dir=~/simultaneous_translation/sinkhorn-simultrans/checkpoints/$TASK

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train \
    --max-tokens 8000 \
    --update-freq 1 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch causal_encoder_iwslt_de_en --delay ${DELAY} \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.00025 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-mustc-enzh \
    --save-dir $save_dir \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 20 \
    --log-format simple --log-interval 100 \
    --num-workers 4 \
    --seed 1 \
    --fp16
