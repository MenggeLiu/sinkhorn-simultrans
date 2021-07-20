#!/usr/bin/env bash
TASK=teacher_iwslt14_deen
SPLIT=test
EXP=../expiwslt
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
AVG=true
RESULT=${TASK}.$(basename $(dirname $(dirname ${DATA})))

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 5 --lenpen 1.5 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"

python -m fairseq_cli.generate ${DATA} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 8000 --fp16 \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}