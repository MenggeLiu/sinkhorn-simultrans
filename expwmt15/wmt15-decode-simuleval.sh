set -e

USERDIR=~/simultaneous_translation/sinkhorn-simultrans/simultaneous_translation


delay=13
port=12351

CHECKPOINT=/home/liumengge/simultaneous_translation/sinkhorn-simultrans/checkpoints_wmt15/sinkhorn_delay${delay}_ft/checkpoint_best.pt

SRC=de
TGT=en
test_name=test
AGENT=/home/liumengge/simultaneous_translation/sinkhorn-simultrans/eval/agents/simul_t2t_ctc_mgliu.py
SRC_FILE=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep/${test_name}.$SRC
TGT_FILE=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep/${test_name}.$TGT
mustc_data_raw=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep
mustc_data_bin=~/simultaneous_translation/simul_confi/data_bins/deen_wmt15

for test_waitk in 1 3 5 7 9 11 13 200; do

OUTPUT=/home/liumengge/simultaneous_translation/sinkhorn-simultrans/simul_decode/wmt15_deen/$test_name/sinkhorn${delay}/test_wait$test_waitk
mkdir -p ${OUTPUT}

WORKERS=2
BLEU_TOK=13a
UNIT=word
if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
  NO_SPACE="--no-space"
fi

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id simuleval \
  --src $SRC --tgt $TGT \
  --src_bpe_code $mustc_data_raw/bpe.32000.en-de \
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
  --port ${port} \
  --workers ${WORKERS} --gpu \
  --test-waitk $test_waitk

done
