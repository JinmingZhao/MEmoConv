# 直接finetune的时候 --model_name_or_path  
# bert-base-chinese
# hfl/chinese-bert-wwm-ext
# hfl/chinese-roberta-wwm-ext

export PYTHONPATH=/data9/MEmoConv

gpuid=$1
output_dir=/data9/memoconv/results/utt_baseline/bert_finetune
bert_data_dir=/data9/memoconv/modality_fts/utt_baseline/

for lr in 2e-5 3e-5 5e-5
do
    CUDA_VISIBLE_DEVICES=${gpuid} python run_cls.py \
        --corpus_name chmed \
        --model_name_or_path hfl/chinese-roberta-wwm-ext \
        --train_file ${bert_data_dir}/train/bert_data.csv \
        --validation_file ${bert_data_dir}/val/bert_data.csv \
        --test_file ${bert_data_dir}/test/bert_data.csv \
        --max_length 20 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 6 \
        --patience 2 \
        --learning_rate ${lr} \
        --lr_scheduler_type 'linear' \
        --output_dir ${output_dir}/roberta_wwm_base_chinese_lr${lr}_bs32/
        # --output_dir ${output_dir}/bert_wwm_base_chinese_lr${lr}_bs32/
        # --output_dir ${output_dir}/bert_base_chinese_lr${lr}_bs32/
done