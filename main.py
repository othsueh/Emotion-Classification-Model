import torch
import argparse
from utils import config, experiments_config
from experiment import run_experiment
from test import run_test

def main():
    
    parser = argparse.ArgumentParser(description='Run experiments or tests')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    assert torch.cuda.is_available(), "CUDA is not available. Please use a GPU to run this code."

    exp_config = experiments_config['base_config']
    experiments = experiments_config['experiments']
    exp_count = 1
    print(f"Run {len(experiments)} experiments")
    
    for exp in experiments:
        if exp['config'] != 'base_config':
            exp_config = exp['config']
        if 'config_update' in exp:
            exp_config.update(exp['config_update']) 
        print('='*30 + f"Experiment {exp_count}: {exp['name']}" + '='*30)

        if args.test:
            # run_test(exp['model_type'], **exp_config)
            run_test(exp['model_type'],**exp_config)
            print("Hi")
        else:
            run_experiment(exp['model_type'], **exp_config)
        exp_count += 1

if __name__ == "__main__":
    main()