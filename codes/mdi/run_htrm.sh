source activate mdi

gpuid=$1
result_dir=/data9/MEmoConv/memoconv/results/mdi
feat_path=/data9/MEmoConv/memoconv/modality_fts/dialogrnn/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl

for run_idx in 1 2 3;
do
  CUDA_VISIBLE_DEVICES=${gpuid} python run_htrm.py \
        --device ${gpuid} --run_idx ${run_idx} \
        --dataset_name M3ED \
        --bert_dim 768  --lr 3e-5 --lr2 3e-5  --epochs 15 \
        --scheduler_type cosine --no_early_stop \
        --use_trm --utr_dim 384 --trm_layers 4 --trm_heads 6 --trm_ff_dim 1024 \
        --attn_type global intra inter local \
        --use_spk_attn --residual_spk_attn \
        --bert_path bert-base-chinese \
        --finetune_layers 4 \
        --bert_frozen \
        --multi_modal --mm_type ecat --modals avl \
        --local_window 3 \
        --same_encoder --use_utt_text_features \
        --result_dir ${result_dir} \
        --feat_path ${feat_path}
done