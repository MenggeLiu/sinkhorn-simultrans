#!/usr/bin/env bash

SRC=de
TGT=en
DATA=~/simultaneous_translation/simul_confi/data_bins/deen_wmt15
WANDB_START_METHOD=thread
FAIRSEQ=~/simultaneous_translation/sinkhorn-simultrans/fairseq
USERDIR=~/simultaneous_translation/sinkhorn-simultrans/simultaneous_translation
PYTHONPATH="$FAIRSEQ:$PYTHONPATH"


DELAY=1
TASK=ctc_delay${DELAY}

save_dir=~/simultaneous_translation/sinkhorn-simultrans/checkpoints_wmt15/$TASK

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train \
    --max-tokens 4000 \
    --update-freq 4 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch causal_encoder --delay ${DELAY} \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.00025 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 200000 \
    --scoring sacrebleu --sacrebleu-tokenizer 13a \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1, "max_len_b": 50}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-wmt15-deen  --find-unused-parameters \
    --save-dir $save_dir \
    --no-epoch-checkpoints \
    --save-interval-updates 2000 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 30 \
    --log-format simple --log-interval 100 \
    --num-workers 8 \
    --seed 1 \
    --fp16
