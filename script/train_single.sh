#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=2-0:00
#SBATCH --job-name=SS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

cd /home/ali/CL-MOLA-Jigsaw

conda activate cjig_env

reg=0.01
lr=1e-4
seed=0


for task in jigsaw-obscene ucc-generalisation_unfair hate-hateful dygen-hate ucc-healthy jigsaw-threat ucc-condescending ucc-hostile ucc-antagonize jigsaw-identity_attack jigsaw-toxicity personal_attack-tpa cad-affiliationdirectedabuse ucc-generalisation ghc-hd hate-offensive abusive-hateful ucc-dismissive personal_attack-a cad-persondirectedabuse jigsaw-insult ucc-sarcastic ghc-vo abusive-abusive personal_attack-ra cad-identitydirectedabuse
do
   echo $task

# # BART-adapter-single
# python run_model.py --output_dir runs/adapter_single_${task}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --no_param_gen \
# --tasks $task --balance_ratio 0.3

# BART-BiHNet-single
# python run_model.py --output_dir runs/BiHNet_single_${task}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 32 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 5000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive \
# --tasks $task --balance_ratio 0.3

# BART single
# python run_model.py --output_dir runs/single_${task}_${reg}_s64_d256_limit/${lr}/${seed} \
# --do_train --eval_period 100000 --eval_at_epoch_end  --wait_step 3 --num_train_epochs 100 --seed ${seed} \
# --train_batch_size 64 --gradient_accumulation_steps 2 --learning_rate ${lr} --max_output_length 8 \
# --generator_hdim 32 --example_limit 100 --train_limit 3000  --h_l2reg ${reg} \
# --adapter_dim 256 --adapter_dim_final 64  --hard_long_term  --limit_label_vocab_space \
# --sample_batch --scale_loss --stm_size 64 --cl_method naive --no_param_gen --skip_adapter --train_all \
# --tasks $task --balance_ratio 0.3

done
