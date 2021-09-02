export PYTHONPATH=/data9/MEmoConv
set -e
modality=$1 # 
gpu=$2
for run_idx in 1 2;
do
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python train_chmed.py --modals=$modality 
        --path AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed
        --run_idx=$run_idx --attention general --active_listener --class_weight 
        --max_epoch 60 --patience 10 --fix_lr_epoch 30 --warmup_epoch 5
        --use_input_project
        --emotion_dim 180 --batch_size 16 --dropout 0.4 --lr 0.0003 --l2 0.0
        --base_model 'LSTM' --graph_model --nodal_attention 
        --windowp 10 --windowf 10
    "
    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done

# 默认的DialgueGCN的配置，只修改了维度和use_input_project