set -e

USERDIR=~/simultaneous_translation/sinkhorn-simultrans/simultaneous_translation

CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt

PORT=12345
WORKERS=2
BLEU_TOK=13a
UNIT=word
DATANAME=$(basename $(dirname $(dirname ${DATA})))
OUTPUT=${DATANAME}_${TGT}-results/${MODEL}.${DATANAME}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
  NO_SPACE="--no-space"
fi

SRC=en
TGT=zh
test_name=tst-COMMON
AGENT=/home/liumengge/simultaneous_translation/sinkhorn-simultrans/eval/agents/simul_t2t_ctc_mgliu.py
SRC_FILE=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh/${test_name}.$SRC
TGT_FILE=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh/${test_name}.$TGT
mustc_data_raw=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh
mustc_data_bin=~/simultaneous_translation/simul_confi/data_bins/mustc_enzh

OUTPUT=/home/liumengge/simultaneous_translation/sinkhorn-simultrans/simul_decode/mustc_enzh/$test_name/best_result_${delta}

simuleval \
  --src $SRC --tgt $TGT \
  --src_bpe_code $mustc_data_raw/bpe.30000.en \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_FILE} \
  --target ${TGT_FILE} \
  --data-bin $mustc_data_bin \
  --model-path ${CHECKPOINT} \
  --output ${OUTPUT} \
  --incremental-encoder \
  --sacrebleu-tokenizer ${BLEU_TOK} \
  --eval-latency-unit ${UNIT} \
  ${NO_SPACE} \
  --scores \
  --port ${PORT} \
  --workers ${WORKERS}