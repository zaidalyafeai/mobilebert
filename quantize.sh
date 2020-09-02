
export DATA_DIR=gs://arabert-mobilebert/qa-data/
export OUTPUT_DIR=gs://arabert-mobilebert/qa-model/
export EXPORT_DIR=gs://arabert-mobilebert/qa-quantized/
python3 run_squad.py \
	--use_quantized_training=true \
	--use_post_quantization=true \
	--activation_quantization=true \
	--data_dir=${DATA_DIR}  \
	--output_dir=${OUTPUT_DIR} \
	--vocab_file=../bert/vocab.txt \
	--bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
	--train_file=../MLQA_V1/dev/dev-context-ar-question-ar.json \
	--export_dir=${EXPORT_DIR} \
