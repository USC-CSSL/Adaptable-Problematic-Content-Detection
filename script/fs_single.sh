#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=1-0:00
#SBATCH --job-name=sfs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

cd /home/ali/CL-MOLA-Jigsaw

conda activate cjig_env
few_shot_num_train_epochs=800
# few_shot_num_train_epochs=1
few_shot_eval_period=200
# few_shot_eval_period=1
postfix=naive_16shot
task_collection=third_set_fs

reg=0.01
lr=1e-4
seed=0


# BART-adapter-single
python run_model.py --do_few_shot_predict --output_dir runs/adapter_single_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
--k_shot 16 --cl_method naive --hard_long_term --no_param_gen  --no_short_term --long_term_task_emb_num 0 \
--eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
--train_batch_size 64 --gradient_accumulation_steps 4 --learning_rate ${lr} --max_output_length 8 \
--example_limit 100 --train_limit 10000  --h_l2reg ${reg} --postfix $postfix \
--adapter_dim 256 --adapter_dim_final 64 --limit_label_vocab_space \
--sample_batch --stm_size 64  --predict_batch_size 16  --scale_by_accumulation  \
--few_shot_wait_step 100000 --few_shot_train_batch_size 16 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --few_shot_eval_period ${few_shot_eval_period} \
--task_collection $task_collection --skip_tasks 20 21 33 --fresh_checkpoint



# # # BART-BiHNet-single
# python run_model.py --do_few_shot_predict --output_dir runs/BiHNet_single_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --k_shot 16 --cl_method naive --hard_long_term --no_short_term  --generator_hdim 32 --long_term_task_emb_num 0 \
# --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 64 --gradient_accumulation_steps 4 --learning_rate ${lr} --max_output_length 8 \
# --example_limit 100 --train_limit 10000  --h_l2reg ${reg} --postfix $postfix \
# --adapter_dim 256 --adapter_dim_final 64 --limit_label_vocab_space \
# --sample_batch --stm_size 64  --predict_batch_size 16  --scale_by_accumulation  \
# --few_shot_wait_step 100000 --few_shot_train_batch_size 16 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --few_shot_eval_period ${few_shot_eval_period} \
# --task_collection $task_collection --skip_tasks 20 21 33 --fresh_checkpoint 


# # BART single
# python run_model.py --do_few_shot_predict --output_dir runs/single_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --k_shot 16 --cl_method naive --hard_long_term --no_param_gen --skip_adapter --train_all --no_short_term --long_term_task_emb_num 0 \
# --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 64 --gradient_accumulation_steps 4 --learning_rate ${lr} --max_output_length 8 \
# --example_limit 100 --train_limit 10000  --h_l2reg ${reg} --postfix $postfix \
# --adapter_dim 256 --adapter_dim_final 64 --limit_label_vocab_space \
# --sample_batch --stm_size 64  --predict_batch_size 16  --scale_by_accumulation  \
# --few_shot_wait_step 100000 --few_shot_train_batch_size 16 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --few_shot_eval_period ${few_shot_eval_period} \
# --task_collection $task_collection --skip_tasks 20 21 33 --fresh_checkpoint 

