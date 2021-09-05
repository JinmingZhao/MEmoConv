export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1

# LA LV AV LAV with 512 and A V L with 256
# AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed :  A LA AV LAV
# Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed : A LA AV LAV
# Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed : V L LV 

cmd="CUDA_VISIBLE_DEVICES=0 python train_chmed.py --modals=LAV
    --path Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed
    --attention general --active_listener --class_weight
    --max_epoch 80 --patience 20 --fix_lr_epoch 30 --warmup_epoch 5
    --dropout 0.1 --rec_dropout 0.1 --lr 5e-4  --l2 0.00001 --batch_size 32
    --use_input_project
    --global_dim 512 --person_dim 512 --emotion_dim 128 --classifer_dim 128 --attention_dim 128
    --is_test --run_idx=3 --best_eval_wf1_epoch 2  --best_eval_f1_epoch 6
"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh