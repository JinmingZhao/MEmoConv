export PYTHONPATH=/data9/MEmoConv
set -e
modality=$1 # 
gpu=$2
for run_idx in 1 2;
do
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python train_chmed.py --modals=$modality 
        --path AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed
        --run_idx=$run_idx --attention general --active_listener --class_weight 
        --max_epoch 80 --patience 20 --fix_lr_epoch 30 --warmup_epoch 5
        --dropout 0.1 --rec_dropout 0.1 --lr 5e-4  --l2 0.00001 --batch_size 32
        --use_input_project
        --global_dim 512 --person_dim 512 --emotion_dim 128 --classifer_dim 128 --attention_dim 128
    "
    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done

# 类别同样不均衡，按照Meld的setting跑 python train_meld.py --active-listener --class-weight --residual --classify sentiment
# attention = 'general'
# class_weight = False
# dropout = 0.1
# rec_dropout = 0.1
# l2 = 0.00001
# lr = 0.0005