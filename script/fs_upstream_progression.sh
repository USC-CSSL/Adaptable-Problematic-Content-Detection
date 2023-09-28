#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=2-0:00
#SBATCH --job-name=FS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1





cd ~/
source clenv/bin/activate
cd Continual-Problematic-Content-Detection-Benchmark
pwd

gpu=6
reg=0.01
lr=1e-4
seed=0

task_collection=third_set_fs
long_term_task_emb_num=26

adapter_vanilla=adapter_vanilla_published_temporal_set_${reg}_s64_d256_limit
BiHNet_vanilla=BiHNet_vanilla_published_temporal_set_${reg}_s64_d256_limit

BiHNet_ewc=BiHNet_ewc_published_temporal_set_${reg}_s64_d256_limit
BiHNet_reg=BiHNet_Reg_published_temporal_set_${reg}_s64_d256_limit

BiHNet_Multitask=mtl_hnet_published_temporal_set_${reg}_s64_d256_limit
Adapter_Multitask=adpter_mtl_published_temporal_set_${reg}_s64_d256_limit

few_shot_num_train_epochs=800
# few_shot_num_train_epochs=1
few_shot_eval_period=200
# few_shot_eval_period=1


for i in {17..22}
do
    filename="best-model_task_$i.pt"
    postfix=naive_16shot_task_$i

    echo "Processing $filename and saving to $postfix for task_collection $task_collection"


    CUDA_VISIBLE_DEVICES=$gpu python run_model.py --do_few_shot_predict --output_dir runs/${adapter_vanilla}/${lr}/${seed} --max_input_length 100  \
    --predict_checkpoint $filename \
    --k_shot 16  --cl_method naive --hard_long_term --no_param_gen --no_short_term \
    --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
    --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
    --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
    --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
    --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
    --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
    --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
    --task_collection $task_collection --skip_tasks 20 21 33 &


    # BART-BiHNet+Reg
    CUDA_VISIBLE_DEVICES=$gpu python run_model.py --do_few_shot_predict --output_dir runs/${BiHNet_reg}/${lr}/${seed} --max_input_length 100  \
    --predict_checkpoint $filename \
    --k_shot 16  --cl_method naive --hard_long_term --no_short_term \
    --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
    --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
    --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
    --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
    --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
    --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
    --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
    --task_collection $task_collection --skip_tasks 20 21 33 &

# # BART-Adapter-Multitask
# python run_model.py --do_few_shot_predict --output_dir runs/${Adapter_Multitask}/${lr}/${seed} --max_input_length 100  \
# --k_shot 16  --cl_method naive --hard_long_term --no_param_gen --no_short_term \
# --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
# --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
# --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
# --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
# --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
# --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
# --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
# --task_collection $task_collection --skip_tasks 20 21 33

# # BART-BiHNet-Multitask
# python run_model.py --do_few_shot_predict --output_dir runs/${BiHNet_Multitask}/${lr}/${seed} --max_input_length 100  \
# --k_shot 16  --cl_method naive --hard_long_term --no_short_term \
# --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
# --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
# --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
# --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
# --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
# --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
# --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
# --task_collection $task_collection --skip_tasks 20 21 33


    # BART-BiHNet-Vanilla
    CUDA_VISIBLE_DEVICES=$gpu python run_model.py --do_few_shot_predict --output_dir runs/${BiHNet_vanilla}/${lr}/${seed} --max_input_length 100  \
    --predict_checkpoint $filename \
    --k_shot 16  --cl_method naive --hard_long_term --no_short_term \
    --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
    --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
    --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
    --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
    --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
    --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
    --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
    --task_collection $task_collection --skip_tasks 20 21 33 &

    # BART-BiHNet+EWC
    CUDA_VISIBLE_DEVICES=$gpu python run_model.py --do_few_shot_predict --output_dir runs/${BiHNet_ewc}/${lr}/${seed} --max_input_length 100  \
    --predict_checkpoint $filename \
    --k_shot 16  --cl_method naive --hard_long_term --no_short_term \
    --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 \
    --seed ${seed} --train_batch_size 64 --predict_batch_size 16 --few_shot_train_batch_size 16 \
    --few_shot_wait_step 100000 --few_shot_num_train_epochs ${few_shot_num_train_epochs} --wait_step 3 --gradient_accumulation_steps 4 \
    --scale_by_accumulation --learning_rate ${lr} --max_output_length 8  --generator_hdim 32 \
    --example_limit 100 --train_limit 10000 --h_l2reg ${reg} --adapter_dim 256 \
    --adapter_dim_final 64 --limit_label_vocab_space --long_term_task_emb_num ${long_term_task_emb_num} \
    --postfix $postfix  --sample_batch --stm_size 64 --few_shot_eval_period ${few_shot_eval_period} \
    --task_collection $task_collection --skip_tasks 20 21 33 
done

