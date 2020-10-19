export TPU_NAME=arabert-mobilebert
export OUTPUT_DIR=gs://arabert-mobilebert/mobilebert-tmp

python3 run_inference.py \
  --bert_config_file=config/mobilebert.json \
  --do_predict \
  --init_checkpoint=gs://arabert-mobilebert/mobilebert-tmp/model.ckpt-10000 \
  --predict_batch_size=8 \
  --output_dir=${OUTPUT_DIR} \
  --data_dir=inference_data \
  --vocab_file=gs://arabert-mobilebert/arabert/vocab.txt \
  --use_summary \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
