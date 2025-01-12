#!/usr/bin/env bash

export SRC=de
export TGT=en
export DATA=/livingrooms/george/wmt15/de-en/data-bin
export WANDB_START_METHOD=thread
export FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate


DELAY=1
TASK=ctc_delay${DELAY}

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_distill_${TGT} \
    --max-tokens 8000 \
    --update-freq 4 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch causal_encoder --delay ${DELAY} \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 300000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-wmt15 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --seed 1 \
    --fp16
