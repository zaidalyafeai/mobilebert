export TEACHER_CHECKPOINT=/path/to/checkpoint/
export OUTPUT_DIR=/tmp/mobilebert/experiment/
export TPU_NAME= arabert-mobilebert

python3 run_pretraining.py \
  --attention_distill_factor=1 \
  --bert_config_file=config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json \
  --bert_teacher_config_file=config/uncased_L-24_H-1024_B-512_A-4.json \
  --beta_distill_factor=5000 \
  --distill_ground_truth_ratio=0.5 \
  --distill_temperature=1 \
  --do_train \
  --first_input_file=gs://arabert-mobilebert \
  --first_max_seq_length=128 \
  --first_num_train_steps=0 \
  --first_train_batch_size=4096 \
  --gamma_distill_factor=5 \
  --hidden_distill_factor=100 \
  --init_checkpoint=${TEACHER_CHECKPOINT} \
  --input_file=path/to/pretraining_data \
  --layer_wise_warmup \
  --learning_rate=0.0015 \
  --max_predictions_per_seq=20 \
  --max_seq_length=512 \
  --num_distill_steps=240000 \
  --num_train_steps=500000 \
  --num_warmup_steps=10000 \
  --optimizer=lamb \
  --output_dir=${OUTPUT_DIR} \
  --save_checkpoints_steps=10000 \
  --train_batch_size=2048 \
  --use_einsum \
  --use_summary \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
