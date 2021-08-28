# 由于语音是中文的最好是采用中文的 wav2vec 模型
# using transformers4.4 env 
# 1. facebook/wav2vec2-base
# 2. jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn

export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1
run_idx=$2

cmd="python train_baseline.py --dataset_mode=chmed_wav2vec
    --model=wav2vec_dnn --gpu_ids=$gpu
    --output_dir=/data9/memoconv/results/utt_baseline/wav2vec_finetune
    --print_freq=100 --run_idx=${run_idx}
    --output_dim=7 --cls_layers=256,128 --embd_method=last
    --niter=10 --niter_decay=5 
    --beta1=0.9 --init_type=normal --init_gain=0.02
    --batch_size=8 --lr=3e-5
    --suffix=run${run_idx}
    --wav2vec_name=jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

### for extracting features
# cmd="python extract_ft.py --dataset_mode=chmed_wav2vec
#     --model=wav2vec_dnn --gpu_ids=$gpu
#     --serial_batches
#     --output_dir=/data9/memoconv/results/utt_baseline/wav2vec_finetune
#     --print_freq=100 --run_idx=${run_idx}
#     --output_dim=7 --cls_layers=128,128 --embd_method=last
#     --niter=5 --niter_decay=5 
#     --beta1=0.9 --init_type=normal --init_gain=0.02
#     --batch_size=8 --lr=2e-5
#     --suffix=${embd_method}_run${run_idx}
#     --wav2vec_name=jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
# "
# echo "\n-------------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
# echo "-------------------------------------------------------------------------------------\n"
# echo $cmd | sh