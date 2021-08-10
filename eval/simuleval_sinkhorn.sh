#!/usr/bin/env bash
DELAY=$1
TASK=ctc_delay${DELAY}
AGENT=./agents/simul_t2t_sinkhorn.py
EXP=../expcwmt
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram32000
SRC_FILE=/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${SRC}
TGT_FILE=/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${TGT}.1
# SRC_FILE=debug/tiny.en
# TGT_FILE=debug/tiny.zh
BLEU_TOK=13a
UNIT=word
OUTPUT=${TASK}.$(basename $(dirname $(dirname ${DATA})))

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
  NO_SPACE="--no-space"
fi

simuleval \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_FILE} \
  --target ${TGT_FILE} \
  --data-bin ${DATA} \
  --model-path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --src-splitter-path ${SPM_PREFIX}_${SRC}.model \
  --tgt-splitter-path ${SPM_PREFIX}_${TGT}.model \
  --output ${OUTPUT} \
  --incremental-encoder \
  --sacrebleu-tokenizer ${BLEU_TOK} \
  --eval-latency-unit ${UNIT} \
  --segment-type ${UNIT} \
  ${NO_SPACE} \
  --scores \
  --test-waitk ${DELAY} \
  --non-strict

mv ${OUTPUT}/scores ${OUTPUT}/scores.$2