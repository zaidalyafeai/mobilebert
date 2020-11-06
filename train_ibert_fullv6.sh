export OUTPUT_DIR=gs://arabert-mobilebert/ibert-sixth
export TPU_NAME=arabert-mobilebertv2
export DATA=gs://arabert-mobilebert/**/tf_examples.tfrecord*
python3 run_pretraining.py \
  --first_input_file=${DATA} \
  --input_file=${DATA} \
  --output_dir=${OUTPUT_DIR} \
  --do_train=True \
  --bert_config_file=config/ibert.json \
  --first_num_train_steps=0 \
  --num_train_steps=2000000 \
  --first_train_batch_size=4096 \
  --train_batch_size=4096 \
  --eval_batch_size=128 \
  --max_seq_length=512 \
  --save_checkpoints_steps=10000 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=20 \
  --learning_rate=0.005 \
  --optimizer=lamb \
  --use_einsum=True \
  --use_summary=True \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
