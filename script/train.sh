#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=2-0:00
#SBATCH --job-name=M
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


cd ../
source venv/bin/activate

reg=0.01
lr=1e-4
seed=0

task_collection=$1
eval_every_k_tasks=25
mtl_task_num=26

pretrianed_model='xlm-roberta-base'

# models=("BiHNet-Reg" "BiHNet-EWC" "Adapter-Vanilla" "BiHNet-Vanilla" "Adapter-Multitask" "BiHNet-Multitask")
models=("BiHNet-Reg")
gpus=(6)

total_runs=$((${#models[@]}))
current_run=1
for model_index in "${!models[@]}"; do
    gpu=${gpus[$model_index]}
    model=${models[$model_index]}
    
    echo "Running $model, task_collection: $task_collection, gpu: $gpu ($current_run/$total_runs)"
    SESSION_NAME="${model}_${task_collection}_gpu_${gpu}"

    if [ "$model" = "BiHNet-Reg" ]; then
        screen -dmS "$SESSION_NAME" bash -c "
        CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_Reg_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --model ${pretrianed_model} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method hnet --eval_every_k_tasks ${eval_every_k_tasks} \
                                                                        --task_collection $task_collection --balance_ratio 0.3 
                                                                        ; exit "
    elif [ "$model" = "BiHNet-EWC" ]; then
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_ewc_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --model ${pretrianed_model} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 16 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term --eval_every_k_tasks ${eval_every_k_tasks} \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method ewc \
                                                                        --task_collection $task_collection --balance_ratio 0.3 ; exit"
    elif [ "$model" = "Adapter-Vanilla" ]; then
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/adapter_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --model ${pretrianed_model} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method naive --no_param_gen --eval_every_k_tasks ${eval_every_k_tasks} \
                                                                        --task_collection $task_collection  --balance_ratio 0.3 ; exit"
    elif [ "$model" = "BiHNet-Vanilla" ]; then
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --model ${pretrianed_model} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method naive --eval_every_k_tasks ${eval_every_k_tasks} \
                                                                        --task_collection $task_collection --balance_ratio 0.3 ; exit"
    elif [ "$model" = "Adapter-Multitask" ]; then
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python run_model.py  \
                                                                        --model ${pretrianed_model} \
                                                                        --output_dir runs/adpter_mtl_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --no_param_gen --mtl_task_num ${mtl_task_num} \
                                                                        --task_collection $task_collection --balance_ratio 0.3 ; exit"
    elif [ "$model" = "BiHNet-Multitask" ]; then
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python run_model.py  \
                                                                        --model ${pretrianed_model} \
                                                                        --output_dir runs/mtl_hnet_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
                                                                        --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
                                                                        --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
                                                                        --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
                                                                        --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
                                                                        --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --mtl_task_num ${mtl_task_num} \
                                                                        --task_collection $task_collection --balance_ratio 0.3 ; exit"
    else
        echo "Invalid model"
    fi

    ((current_run++))
done
# # BiHNet+Reg
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_Reg_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method hnet --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection --balance_ratio 0.3

# # # BiHNet+EWC
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_ewc_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 16 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term --eval_every_k_tasks ${eval_every_k_tasks} \
# --sample_batch --scale_loss --stm_size 64 --cl_method ewc \
# --task_collection $task_collection --balance_ratio 0.3

# # Adapter-Vanilla
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/adapter_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --no_param_gen --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection  --balance_ratio 0.3

# # BiHNet+Vanilla
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py --output_dir runs/BiHNet_vanilla_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --eval_every_k_tasks ${eval_every_k_tasks} \
# --task_collection $task_collection --balance_ratio 0.3

# # Adapter-Multitask
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py  \
# --output_dir runs/adpter_mtl_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --no_param_gen --mtl_task_num ${mtl_task_num} \
# --task_collection $task_collection --balance_ratio 0.3

# # BiHNet-Multitask
# CUDA_VISIBLE_DEVICES=$gpu python run_model.py  \
# --output_dir runs/mtl_hnet_${task_collection}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --mtl --mtl_task_num ${mtl_task_num} \
# --task_collection $task_collection --balance_ratio 0.3
