# eg:
# bash run.sh V 0 1
export PYTHONPATH=/data7/MEmoConv
set -e
modality=$1 # 
gpu=$2
run_ind=$3
for run_idx in $run_ind;
do
    cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
        --dataset_mode chmed
        --pretained_ft_type utt_baseline
        --num_threads 0 --run_idx=$run_idx
        --dropout_rate 0.5  --learning_rate 1e-3 --batch_size 64 --postfix self
        --v_ft_type denseface --v_input_size 342 --max_text_tokens 50
        --a_ft_type wav2vec --a_input_size 768 --max_acoustic_tokens 128 
        --l_ft_type bert_base_chinese --l_input_size 768 --max_visual_tokens 64 
        --l_hidden_size 128 --v_hidden_size 128 --a_hidden_size 128 --mid_fusion_layers '256,128'
    "
    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done