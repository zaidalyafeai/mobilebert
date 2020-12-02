!python3 run_pretraining.py \
  --attention_distill_factor=1 \
  --bert_config_file=config/mobilebert.json \
  --bert_teacher_config_file=config/ibert-small-50K.json \
  --beta_distill_factor=5000 \
  --distill_ground_truth_ratio=0.5 \
  --distill_temperature=1 \
  --do_train \
  --gamma_distill_factor=5 \
  --hidden_distill_factor=100 \
  --init_checkpoint=gs://arabert-mobilebert/ibert-9/model.ckpt-1600000 \
  --input_file=gs://arabert-mobilebert/dataset-50K/train* \
  --layer_wise_warmup \
  --learning_rate=2e-5 \
  --max_predictions_per_seq=20 \
  --max_seq_length=512 \
  --num_distill_steps=240000 \
  --num_train_steps=2000000 \
  --num_warmup_steps=10000 \
  --optimizer=lamb \
  --output_dir=gs://arabert-mobilebert/mobilebert-160K-15e-4 \
  --save_checkpoints_steps=10000 \
  --train_batch_size=256 \
  --use_einsum \
  --use_summary \
  --use_tpu \
  --tpu_name=arabert-mobilebertv1  \