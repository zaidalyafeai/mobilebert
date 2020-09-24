
export DATA_DIR=gs://arabert-mobilebert/qa-data/
export OUTPUT_DIR=gs://arabert-mobilebert/qa-model/
export TPU_NAME=arabert-mobilebert
python3 run_squad.py \
	--bert_config_file=config/mobilebert_qa.json \
	--data_dir=${DATA_DIR} \
	--do_train \
	--doc_stride=128 \
	--init_checkpoint=gs://arabert-mobilebert/mobilebert-ckpt/model.ckpt-1000.index \
	--learning_rate=4e-05 \
	--predict_file=../arcd-train.json \
	--do_lower_case \
  	--do_predict \
	--max_answer_length=30 \
	--max_query_length=64 \
	--max_seq_length=384 \
	--n_best_size=20 \
	--num_train_epochs=1 \
	--output_dir=${OUTPUT_DIR} \
	--train_batch_size=16 \
	--train_file=../arcd-train.json \
	--use_tpu \
	--tpu_name=${TPU_NAME} \
	--vocab_file=gs://arabert-mobilebert/arabert/vocab.txt \
	--warmup_proportion=0.1 \
	--verbose_logging=True \
