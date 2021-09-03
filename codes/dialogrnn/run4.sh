export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1

# Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese.pkl

# for modality in A V L;
# do
#     for run_idx in 1 2 3;
#     do
#     cmd="CUDA_VISIBLE_DEVICES=${gpu} python train_chmed.py --modals=$modality 
#         --path Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese
#         --run_idx=$run_idx --attention general --active_listener --class_weight 
#         --max_epoch 80 --patience 20 --fix_lr_epoch 30 --warmup_epoch 5
#         --dropout 0.1 --rec_dropout 0.1 --lr 5e-4  --l2 0.00001 --batch_size 32
#         --use_input_project
#         --global_dim 256 --person_dim 256 --emotion_dim 128 --classifer_dim 128 --attention_dim 128
#     "
#     echo "\n-------------------------------------------------------------------------------------"
#     echo "Execute command: $cmd"
#     echo "-------------------------------------------------------------------------------------\n"
#     echo $cmd | sh
#     done
# done

# for modality in LA LV;
# do
#     for run_idx in 1 2 3;
#     do
#     cmd="CUDA_VISIBLE_DEVICES=${gpu} python train_chmed.py --modals=$modality 
#         --path Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese
#         --run_idx=$run_idx --attention general --active_listener --class_weight 
#         --max_epoch 80 --patience 20 --fix_lr_epoch 30 --warmup_epoch 5
#         --dropout 0.1 --rec_dropout 0.1 --lr 5e-4  --l2 0.00001 --batch_size 32
#         --use_input_project
#         --global_dim 512 --person_dim 512 --emotion_dim 128 --classifer_dim 128 --attention_dim 128
#     "
#     echo "\n-------------------------------------------------------------------------------------"
#     echo "Execute command: $cmd"
#     echo "-------------------------------------------------------------------------------------\n"
#     echo $cmd | sh
#     done
# done

# for modality in AV LAV;
# do
#     for run_idx in 1 2 3;
#     do
#     cmd="CUDA_VISIBLE_DEVICES=${gpu} python train_chmed.py --modals=$modality 
#         --path Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_avg_robert_base_wwm_chinese
#         --run_idx=$run_idx --attention general --active_listener --class_weight 
#         --max_epoch 80 --patience 20 --fix_lr_epoch 30 --warmup_epoch 5
#         --dropout 0.1 --rec_dropout 0.1 --lr 5e-4  --l2 0.00001 --batch_size 32
#         --use_input_project
#         --global_dim 512 --person_dim 512 --emotion_dim 128 --classifer_dim 128 --attention_dim 128
#     "
#     echo "\n-------------------------------------------------------------------------------------"
#     echo "Execute command: $cmd"
#     echo "-------------------------------------------------------------------------------------\n"
#     echo $cmd | sh
#     done
# done

# 类别同样不均衡，按照Meld的setting跑 python train_meld.py --active-listener --class-weight --residual --classify sentiment
# attention = 'general'
# class_weight = False
# dropout = 0.1
# rec_dropout = 0.1
# l2 = 0.00001
# lr = 0.0005