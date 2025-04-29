import torch
from utils import config, experiments_config
from experiment import run_experiment

def main():
    assert torch.cuda.is_available(), "CUDA is not available. Please use a GPU to run this code."

    exp_config = experiments_config['base_config']
    experiments = experiments_config['experiments']
    corpus = exp_config['corpus']
    print(f"Launch total {len(experiments)} experiments with {corpus}")
    exp_count = 1
    for exp in experiments:
        if exp['config'] != 'base_config':
            exp_config = exp['config']
        if 'config_update' in exp:
            exp_config.update(exp['config_update']) 
        print('='*30 + f"Experiment {exp_count}: {exp['name']}" + '='*30)

        run_experiment(exp['model_type'],**exp_config)
        exp_count += 1

if __name__ == "__main__":
    main()