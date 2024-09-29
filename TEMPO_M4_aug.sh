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
for percent_aug in -100 -50 -25 -10 25 50 100; do # 
comment="TimeLLM-M4-aug-${percent_aug}"
  for season in Monthly Yearly Weekly Daily Quarterly Hourly ; do  # 
    python run_m4.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns $season \
      --model $model_name \
      --data m4 \
      --features M \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate $learning_rate \
      --loss 'SMAPE' \
      --train_epochs $train_epochs \
      --model_comment $comment \
      --d_model 768 \
      --n_heads 4 \
      --freq 0 \
      --gpt_layer 3 \
      --is_gpt 1 \
      --model_id m4_TEMPO'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
      --label_len 18 \
      --prompt $prompt\
      --batch_size 256 \
      --decay_fac 0.5 \
      --d_model 768 \
      --d_ff 768 \
      --dropout 0.3 \
      --patch_size 16 \
      --stride 8 \
      --tmax 20 \
      --aug m4-${season}-train_168_gen_50ksteps_10k.npy \
      --percent_aug $percent_aug

  done

done 
# --batch_size 256 \
# --output_dir ./experiments \