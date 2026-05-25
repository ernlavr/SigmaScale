#
python3 run_model_lowrank_replacement.py --input_model "meta-llama/Llama-3.1-8B-Instruct" \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --test_loader_seqlen 2048 \
    --train_loader_seqlen 2048 \
    --test_loader_nsamples 256 \
    --train_loader_nsamples 256 \
