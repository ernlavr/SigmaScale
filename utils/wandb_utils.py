import wandb
import yaml
import os

def parse_sweep_config_from_file(config_path):
    # check if file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Sweep config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    return sweep_config

def start_sweep(sweep_config_path, main_func):
    sweep_config_parsed = parse_sweep_config_from_file(sweep_config_path)
    sweep_id = wandb.sweep(sweep_config_parsed, project='spinquant-noise')
    def sweep_main():
        with wandb.init(settings=wandb.Settings(console='wrap')) as run:
            config = wandb.config
            # Here you can access config parameters and use them in your training
            print(f"Running training with config: {config}")
            main_func()
    wandb.agent(sweep_id, function=sweep_main)
    
def start_run(run_name, config, main_func):
    with wandb.init(project='spinquant-noise', 
                    name=run_name, 
                    config=config, 
                    settings=wandb.Settings(console='wrap')) as run:
        main_func()
        
    
    
def get_sweep_config(sweep_parameters):
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'perplexity',
            'goal': 'minimize'
        },
        'parameters': sweep_parameters
    }
    return sweep_config