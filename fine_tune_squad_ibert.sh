export OUTPUT_DIR=gs://arabert-mobilebert/ibert-first
export TPU_NAME=arabert-mobilebert
python run_squad.py \
  --vocab_file=../bert/bert-vocab.txt \
  --bert_config_file=config/ibert.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-50000.index \
  --do_train=True \
  --train_file=../arcd-train.json\
  --do_predict=True \
  --predict_file=../arcd-train.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://arabert-mobilebert/tmp/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME

python evaluate-v1.1.py ../arcd-train.json ../bucket/tmp/predictions.json