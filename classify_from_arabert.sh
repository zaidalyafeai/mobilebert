
export DATA_DIR=gs://arabert-mobilebert/cs-data/
export OUTPUT_DIR=gs://arabert-mobilebert/cs-model/
export TPU_NAME=arabert-mobilebert
python3 run_classifier.py \
	--bert_config_file=config/mobilebert_qa.json \
	--task_name=sst-2\
	--data_dir=${DATA_DIR} \
	--do_train \
	--init_checkpoint=gs://arabert-mobilebert/mobilebert-ckpt/model.ckpt-470000 \
	--learning_rate=4e-05 \
  	--do_eval \
	--max_seq_length=384 \
	--num_train_epochs=1 \
	--output_dir=${OUTPUT_DIR} \
	--train_batch_size=64 \
	--data_dir=../hard \
	--use_tpu \
	--tpu_name=${TPU_NAME} \
	--vocab_file=gs://arabert-mobilebert/arabert/vocab.txt \
	--warmup_proportion=0.1 \
	--verbose_logging=True \
	--use_quantized_training=true

