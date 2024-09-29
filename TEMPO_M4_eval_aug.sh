model_name=TEMPO

train_epochs=100
llama_layers=32
batch_size=24
learning_rate=0.001
d_model=8
d_ff=32

master_port=00097
num_process=8

prompt=1
for percent_aug in -100 -50 -25 25 50 100; do # 100 75 50 25 10 
comment="TimeLLM-M4-aug-${percent_aug}"
  for season in Weekly Daily Quarterly Hourly Monthly Yearly ; do  
    python eval_m4.py \
      --root_path ./dataset/m4 \
      --model $model_name \
      --model_comment $comment 
  done

done 
# --batch_size 256 \
# --output_dir ./experiments \