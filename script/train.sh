#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=2-0:00
#SBATCH --job-name=M
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


cd /home/ali/CL-MOLA-Jigsaw

conda activate cjig_env

reg=0.01
lr=1e-4
seed=0

task_collection=third_set
eval_every_k_tasks=25
mtl_task_num=26

# # BART-BiHNet+Reg
# python run_model.py --output_dir runs/BiHNet_Reg_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method hnet --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection --balance_ratio 0.3

# # BART-BiHNet+EWC
# python run_model.py --output_dir runs/BiHNet_ewc_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 16 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term --eval_every_k_tasks ${eval_every_k_tasks} \
# --sample_batch --scale_loss --stm_size 64 --cl_method ewc \
# --task_collection $task_collection --balance_ratio 0.3

# # BART-Adapter-Vanilla
# python run_model.py --output_dir runs/adapter_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --no_param_gen --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection  --balance_ratio 0.3

# # BART-BiHNet+Vanilla
# python run_model.py --output_dir runs/BiHNet_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection --balance_ratio 0.3

# # BART-Adapter-Multitask
# python run_model.py  \
# --output_dir runs/adpter_mtl_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --no_param_gen --mtl_task_num ${mtl_task_num} \
# --task_collection $task_collection --balance_ratio 0.3

# # BART-BiHNet-Multitask
# python run_model.py  \
# --output_dir runs/mtl_hnet_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --mtl_task_num ${mtl_task_num} \
# --task_collection $task_collection --balance_ratio 0.3
