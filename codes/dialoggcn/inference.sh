export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1

# Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed A V L LA LV AV LAV
# Asent_avg_wav2vev_zh-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed A V L LA LV AV LAV

cmd="CUDA_VISIBLE_DEVICES=2 python train_chmed.py --modals=LAV
    --path Asent_avg_wav2vev_zh-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed
    --attention general --active_listener --class_weight 
    --max_epoch 60 --patience 10 --fix_lr_epoch 30 --warmup_epoch 5
    --use_input_project
    --emotion_dim 180 --batch_size 16 --dropout 0.4 --lr 0.0003 --l2 0.0
    --base_model 'LSTM' --graph_model --nodal_attention 
    --windowp 10 --windowf 10
    --is_test --run_idx=1 --best_eval_wf1_epoch=27 --best_eval_f1_epoch=19
"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

# 默认的DialgueGCN的配置，只修改了维度和use_input_project