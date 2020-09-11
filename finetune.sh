
export DATA_DIR=gs://arabert-mobilebert/qa-data/
export OUTPUT_DIR=gs://arabert-mobilebert/qa-model/
export TPU_NAME=arabert-mobilebert
python3 run_squad.py \
	--bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
	--data_dir=${DATA_DIR} \
	--do_train \
	--doc_stride=128 \
	--init_checkpoint=gs://arabert-mobilebert/mobilebert/mobilebert_variables.ckpt.index \
	--learning_rate=4e-05 \
	--predict_file=../dev-v1.1.json \
	--do_lower_case \
  	--do_predict \
	--max_answer_length=30 \
	--max_query_length=64 \
	--max_seq_length=384 \
	--n_best_size=20 \
	--num_train_epochs=1 \
	--output_dir=${OUTPUT_DIR} \
	--train_batch_size=32 \
	--train_file=../train-v1.1.json \
	--use_tpu \
	--tpu_name=${TPU_NAME} \
	--vocab_file=gs://arabert-mobilebert/mobilebert/vocab.txt \
	--warmup_proportion=0.1 \
	--use_quantized_training=true \
	--verbose_logging=True \