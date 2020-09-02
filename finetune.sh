
export DATA_DIR=gs://arabert-mobilebert/qa-data/
export OUTPUT_DIR=gs://arabert-mobilebert/qa-model/
export TPU_NAME=arabert-mobilebert
export EXPORT_DIR=gs://arabert-mobilebert/qa-quantized/
python3 run_squad.py \
	--bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
	--data_dir=${DATA_DIR} \
	--do_train \
	--doc_stride=128 \
	--init_checkpoint=gs://arabert-mobilebert/ckpt/model.ckpt-1000.index \
	--learning_rate=4e-05 \
	--max_answer_length=30 \
	--max_query_length=64 \
	--max_seq_length=128 \
	--n_best_size=20 \
	--num_train_epochs=5 \
	--output_dir=${OUTPUT_DIR} \
	--train_batch_size=32 \
	--train_file=../MLQA_V1/dev/dev-context-ar-question-ar.json \
	--use_tpu \
	--tpu_name=${TPU_NAME} \
	--vocab_file=../bert/vocab.txt \
	--warmup_proportion=0.1 \
	--use_quantized_training=true \
	--verbose_logging=True
