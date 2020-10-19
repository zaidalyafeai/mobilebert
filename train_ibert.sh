export OUTPUT_DIR=gs://arabert-mobilebert/ibert-ckpt
export TPU_NAME=arabert-mobilebert
export DATA=gs://arabert-mobilebert/**/tf_examples.tfrecord*
python3 run_pretraining.py \
  --input_file=${DATA} \
  --output_dir=${OUTPUT_DIR} \
  --do_train=True \
  --bert_config_file=config/ibert.json \
  --train_batch_size=128 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --save_checkpoint_steps=5000 \
  --num_train_steps=500 \
  --iterations_per_loop=500 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5 \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
  --use_einum \
  --use_summary 
