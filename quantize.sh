
export DATA_DIR=gs://arabert-mobilebert/qa-data/
export OUTPUT_DIR=gs://arabert-mobilebert/qa-model/
export EXPORT_DIR=gs://arabert-mobilebert/qa-quantized/
python3 run_squad.py \
	--use_post_quantization=true \
	--activation_quantization=false \
	--data_dir=${DATA_DIR}  \
	--output_dir=${OUTPUT_DIR} \
	--vocab_file=../bert/en-vocab.txt \
	--bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT_QAT.json \
	--train_file=dev-context-en-question-en.json \
	--export_dir=${EXPORT_DIR} \
