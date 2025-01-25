import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn   
import torch.optim as optim
import random
from torch.utils.data import DataLoader

from utils import *
from display import *
import net

# Parameter Setting
corpus = "MSPPODCAST"
text_feature_extractor = 'roberta-large-UTT'
audio_feature_extractor = 'whisper-large-v3-UTT'
input_dim = 2304
model_name = ''

# Model setup
model = net.SimpleModel(input_dim, dropout=0.3).cuda()
model = model.load_state_dict(torch.load(os.path.join(config['PATH_TO_CKPT'], model_name)))
net_name = model.__class__.__name__

# Set file unique header
file_uni_name = f'{corpus}_{net_name}_{text_feature_extractor}_{audio_feature_extractor}'

def data_loading():
    # Load corpus data
    corpus_path = config[corpus]["PATH_TO_TEST"]
    corpus_df = pd.read_csv(corpus_path)
    return corpus_df

def evaluate(test_loader, corpus_df):

    length = len(test_loader)
    
    # Pre-allocate lists to store predictions and names
    all_preds = []
    all_names = []
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            name = data["name"]
            text, audio = data["text"], data["audio"]
            total_input = torch.cat((text, audio), dim=1)
            total_input = total_input.cuda()

            output = model(total_input)
            pred = output.argmax(dim=1).cpu()
            
            # Store predictions and names
            all_preds.extend(pred)
            all_names.extend(name)
        
            progress_bar(idx, length, f'Predicting {name} -> {indexToFullEmotion(pred)}')
        # Convert predictions to emotion codes in one go
        pred_codes = [indexToEmotion(p) for p in all_preds]
        
        # Create a dictionary for faster lookup
        pred_dict = dict(zip(all_names, pred_codes))

        # Update dataframe in one operation
        corpus_df['EmoClass'] = corpus_df['FileName'].map(pred_dict)
        print(f"Predictions completed for {length} samples")

if __name__ == "__main__":
    corpus_df = data_loading()
    
    # Get feature directories
    text_feature = get_feature_dir(corpus,text_feature_extractor)
    audio_feature = get_feature_dir(corpus,audio_feature_extractor)

    # Create datasets
    test_dataset = MSPTestset(corpus_df, text_feature, audio_feature)
    print(f"Number of training samples: {len(test_dataset)}")

    # Create dataloaders
    test_loader = DataLoader(test_dataset)
    
    # Start timing
    start_time = time.time()
    
    numel_list = [p.numel() for p in model.parameters()]
    total_params = sum(numel_list)
    print(f"Total number of downstream model parameters: {total_params}")

    # evaluate.. 
    evaluate(test_loader, corpus_df)

    corpus_df.to_csv(f"{file_uni_name}_predictions.csv", index=False)
    print(f"Predictions saved to {file_uni_name}_predictions.csv")
    

