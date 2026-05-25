# with KD
python3 run_model_lowrank_replacement.py \
    --input_model "meta-llama/Llama-3.1-8B-Instruct" \
    --compressed_model "/shared/elavrin/SpinQuant/output_dir/saved_models/sweeps/sweep_gp59ed9g/meta-llama_Llama-3.1-8B-Instruct_3dvdhodm_0.75" \
    --use_sensitivity_cache='output_dir/precomputed/meta-llama_Llama-3.1-8B-Instruct_calib_sensitivity_ppl.pt' \
    --access_token '<HF_TOKEN>' \
    --fine_tune_after_compression \
    --train_data=alpaca \
    --use_distillation \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --test_loader_seqlen 2048 \
    --train_loader_seqlen 2048 \
    --test_loader_nsamples 256 \
    --train_loader_nsamples 256 \

# w/o KD
python3 run_model_lowrank_replacement.py \
    --input_model "meta-llama/Llama-3.1-8B-Instruct" \
    --compressed_model "/shared/elavrin/SpinQuant/output_dir/saved_models/sweeps/sweep_gp59ed9g/meta-llama_Llama-3.1-8B-Instruct_3dvdhodm_0.75" \
    --use_sensitivity_cache='output_dir/precomputed/meta-llama_Llama-3.1-8B-Instruct_calib_sensitivity_ppl.pt' \
    --access_token '<HF_TOKEN>' \
    --fine_tune_after_compression \
    --model_max_length 2048 \
    --train_data=alpaca \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --test_loader_seqlen 2048 \
    --train_loader_seqlen 2048 \
    --test_loader_nsamples 256 \
    --train_loader_nsamples 256 \
