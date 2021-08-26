export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1
run_idx=$2

cmd="python train_baseline.py --dataset_mode=chmed_wav2vec
    --hidden_size=256  --model=wav2vec_dnn --gpu_ids=$gpu
    --output_dir=/data9/memoconv/results/utt_baseline/wav2vec_finetune
    --print_freq=100 --run_idx=${run_idx} --suffix=run_${run_idx}
    --output_dim=7 --cls_layers=128,128 --embd_method=last
    --niter=5 --niter_decay=5  
    --beta1=0.9 --init_type=normal --init_gain=0.02
    --batch_size=24 --lr=2e-5
    --name= --suffix={embd_method}_run{run_idx}
    --wav2vec_name=facebook/wav2vec2-base
"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh