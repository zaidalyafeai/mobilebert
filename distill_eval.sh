export OUTPUT_DIR=gs://arabert-mobilebert/mobilebert-ckptv7
export TPU_NAME=arabert-mobilebert
export DATA=gs://arabert-mobilebert/poems/tf_examples.tfrecord
python3 run_pretraining.py \
  --bert_config_file=config/mobilebert.json \
  --do_eval \
  --eval_batch_size=8 \
  --input_file=${DATA} \
  --max_eval_steps=100 \
  --iterations_per_loop=100 \
  --output_dir=${OUTPUT_DIR} \
  --use_einsum \
  --use_summary \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
