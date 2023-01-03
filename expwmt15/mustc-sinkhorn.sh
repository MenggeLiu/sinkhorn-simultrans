#!/usr/bin/env bash
SRC=en
TGT=zh
DATA=~/simultaneous_translation/simul_confi/data_bins/mustc_${SRC}${TGT}
WANDB_START_METHOD=thread
FAIRSEQ=~/simultaneous_translation/sinkhorn-simultrans/fairseq
USERDIR=~/simultaneous_translation/sinkhorn-simultrans/simultaneous_translation
PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

DELAY=$1
TASK=sinkhorn_delay${DELAY}_ft

save_dir=~/simultaneous_translation/sinkhorn-simultrans/checkpoints/$TASK

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --load-pretrained-encoder-from ~/simultaneous_translation/sinkhorn-simultrans/checkpoints/ctc_delay${DELAY}/checkpoint_best.pt \
    --train-subset train \
    --max-tokens 4000 \
    --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch sinkhorn_encoder_iwslt_de_en --delay ${DELAY} --mask-ratio 0.5 \
    --sinkhorn-iters 16 --sinkhorn-tau 0.13 --sinkhorn-noise-factor 0.45 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_ctc --eos-loss --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.00025 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --scoring sacrebleu --sacrebleu-tokenizer ja-mecab \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1, "max_len_b": 50}' \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn-wmt15 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 1000 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 20 \
    --log-format simple --log-interval 100 \
    --num-workers 4 \
    --seed 66 \
    --fp16
