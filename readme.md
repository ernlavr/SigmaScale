# To run
See `scripts/` directory. Run the scripts in the following order:
1. `./compute_sensitivities.sh` - computes the sensitivity of each layer and saves it to a file, output should be saved in `output_dir/low_rank_analysis/cache/<MODEL_NAME>_calib_sensitivity_ppl.pt`
2. `./train_scaling_matrices.sh` - Uses the sensitivity computed in Step 1 to train the scaling matrices for each layer. Output should be saved in `output_dir/low_rank_analysis/saved_models/...`
3. `./finetune_compressed.sh` - Finetunes the compressed model obtained from Step 2 for a 1 epoch to recover some of the lost performance from compression.
    - You can change `--train_data=alpaca` to any other dataset you want to finetune on.