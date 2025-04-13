import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn   
import torch.optim as optim
import random
import wandb
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch.utils.data import DataLoader

from utils import *
from display import *
import net

# Parameter Setting
corpus = "MSPPODCAST"
text_feature_extractor = 'roberta-large-UTT'
audio_feature_extractor = 'whisper-large-v3-UTT'
seed = 42
batch = 64
epoch = 3
input_dim = 2304
audio_dim = 1280
text_dim = 1024
leanring_rate = 1e-3
dropout = 0.2
emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
sample_per_class = [6716,6400,16652,2992,1134,1419,2519,29144]
# Model setup
model = net.MOE(audio_dim,text_dim,dropout=0.2,fusion_type='bilinear',num_experts=3).cuda()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
net_name = model.__class__.__name__

# loss criterion
criterion = net.DiverseExpertLoss(sample_per_class)
optimizer = optim.Adam(model.parameters(), lr=leanring_rate)

# Set file unique header
file_uni_name = f'{corpus}_{net_name}_{text_feature_extractor}_{audio_feature_extractor}'

def data_preprocessing():
    # Load corpus data
    corpus_path = config[corpus]["PATH_TO_LABEL"]
    corpus_df = pd.read_csv(corpus_path)
    corpus_df["FileName"]= corpus_df["FileName"].str.replace('.wav', '')

    # Remove non consensus labels
    main_corpus_df = corpus_df[~corpus_df["EmoClass"].isin(["X", "O"])]

    return main_corpus_df

def train(train_loader):

    length = train_dataset.total_samples // batch

    # Statistics for training performance
    train_loss = 0
    running_macro_f1 = 0
    running_acc = 0
    
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):

        text, audio = data["text"], data["audio"]
        category, avd = label["category"], label["avd"] # avd not use in current model
        
        # Input data
        # total_input = torch.cat((text, audio), dim=1) # Concatenate text and audio features
        true_label = torch.argmax(category, dim=1)
        text, audio, true_label= text.cuda(), audio.cuda(), true_label.cuda()

        output = model(audio,text)

        #! For MOE model only(Need to Modify!)
        extra_info = {}
        logits = output['logits']
        extra_info.update({
            "logits": logits.transpose(0,1)
        })
        output = output['output']
        # Loss
        loss = criterion(output_logits = output, target = true_label, extra_info=extra_info)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # For loss calculation
        train_loss += loss.item()

        # For F1 score, accuracy, and UAR calculation
        pred = output.argmax(dim=1).cpu()
        true_label = torch.argmax(category, dim=1).cpu()
        batch_f1 = f1_score(true_label, pred, average='macro')
        batch_acc = accuracy_score(true_label, pred)
        batch_uar = recall_score(true_label, pred, average='macro',zero_division=0.0)  # UAR is the same as macro-averaged recall
        running_macro_f1 += batch_f1
        running_acc += batch_acc
        running_uar = 0 if batch_idx == 0 else running_uar
        running_uar += batch_uar
        
        progress_bar(batch_idx+1, length, 'Loss: %.3f | Macro F1: %.3f | Acc: %.3f | UAR: %.3f' % 
                    (train_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), 
                     running_acc/(batch_idx+1), running_uar/(batch_idx+1)))
        
    # Calculate average metrics
    avg_loss = train_loss / (batch_idx + 1)
    avg_macro_f1 = running_macro_f1 / (batch_idx + 1) 
    avg_acc = running_acc / (batch_idx + 1)
    avg_uar = running_uar / (batch_idx + 1)
    return avg_loss, avg_macro_f1, avg_acc, avg_uar

def log_view_table(audios, predicted, labels, arousal_valence, probs):
    columns = ["Audio", "Predict", "Target", "Arousal", "Valence"] + emotions
    table = wandb.Table(columns=columns)
    for audio, pred, tar, aro_val, prob in zip(audios,predicted,labels,arousal_valence, probs):
        table.add_data(wandb.Audio(audio,sample_rate=16000),indexToFullEmotion(pred),indexToFullEmotion(tar),aro_val[0],aro_val[1],*prob.numpy()) 
    wandb.log({"predictions_table":table}, commit=False)

def evaluate(valid_loader, last=False, log_view=False, batch_list=[0,1,2,3]):

    length = valid_dataset.total_samples // batch
    
    # Statistics for validation performance
    val_loss = 0
    running_macro_f1 = 0
    running_acc = 0
    
    # Lists to store predictions and true labels for correlation matrix
    if (last):
        all_preds = []
        all_labels = []
    
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):

            text, audio = data["text"], data["audio"]
            category, avd = label["category"], label["avd"]
            if(log_view):
                origin = data["origin"]

            true_label = torch.argmax(category, dim=1)
            text, audio, true_label= text.cuda(), audio.cuda(), true_label.cuda()

            output = model(audio,text)
            output = output['output']
            loss = criterion(output_logits = output, target = true_label)
            true_label = true_label.cpu()

            # For loss calculation
            val_loss += loss.item()

            # For F1 score and accuracy calculation
            pred = output.argmax(dim=1).cpu()
            batch_f1 = f1_score(true_label, pred, average='macro')
            batch_acc = accuracy_score(true_label, pred)
            batch_uar = recall_score(true_label, pred, average='macro',zero_division=0.0)
            running_macro_f1 += batch_f1
            running_acc += batch_acc
            running_uar = 0 if batch_idx == 0 else running_uar
            running_uar += batch_uar
            
            # Log the view
            if(log_view):
                if(batch_idx in batch_list):
                    log_view_table(origin,pred,true_label,avd.cpu(),output.softmax(dim=1).cpu()) 
            
            # Store predictions and labels for correlation matrix
            if (last):
                all_preds.extend(pred)
                all_labels.extend(true_label)
            
            progress_bar(batch_idx+1, length, 'Loss: %.3f | Macro_F1: %.3f | Acc: %.3f | UAR: %.3f' % 
                        (val_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), running_acc/(batch_idx+1), running_uar/(batch_idx+1)))
    
    # Calculate average metrics
    avg_loss = val_loss / (batch_idx + 1)
    avg_macro_f1 = running_macro_f1 / (batch_idx + 1)
    avg_acc = running_acc / (batch_idx + 1)
    avg_uar = running_uar / (batch_idx + 1)
    
    # Plot correlation matrix
    if (last):
        plot_confusion_matrix(all_preds, all_labels, file_uni_name)
    
    return avg_loss, avg_macro_f1, avg_acc, avg_uar

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Login to wandb
    wandb.login(key=config['WANDBAPI'])

    main_corpus_df = data_preprocessing()
    train_df, val_df = corpus_split(main_corpus_df)
    
    # Get feature directories
    text_feature = get_feature_dir(corpus,text_feature_extractor)
    audio_feature = get_feature_dir(corpus,audio_feature_extractor)

    # Create datasets
    train_dataset = MSPDataset(train_df, text_feature, audio_feature, seed=seed)
    valid_dataset = MSPDataset(val_df, text_feature, audio_feature, origin_path=config[corpus]['PATH_TO_AUDIO'], seed=seed, getOriginWav=True)
    print(f"Number of training samples: {train_dataset.total_samples}")
    print(f"Number of validation samples: {valid_dataset.total_samples}")

    # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=16)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=16)
    train_loader = DataLoader(train_dataset, batch_size=batch*torch.cuda.device_count(), 
                            num_workers=4*torch.cuda.device_count(), pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch*torch.cuda.device_count(), 
                            num_workers=4*torch.cuda.device_count(), pin_memory=True)
    
    # Train and evaluate
    # train_losses = []
    # val_losses = []
    # train_macro_f1 = []
    # val_macro_f1 = []
    # train_acc = []
    # val_acc = []
            
    # Start timing
    start_time = time.time()
    
    numel_list = [p.numel() for p in model.parameters()]
    total_params = sum(numel_list)
    print(f"Total number of downstream model parameters: {total_params}")

    wandb.init(
        project="MSP-Podcast",
        tags=["initial Test"], 
        config = {
            "seed": seed,
            "epochs": epoch,
            "learning_rate": leanring_rate,
            "dropout": dropout,
            "batch_size": batch,
            'text_feature_extractor': 'roberta-large-UTT',
            'audio_feature_extractor': 'whisper-large-v3-UTT',
            'input_dim': input_dim
        }
    )
    for e in range(epoch):
        print('='*30+f'Epoch {e+1}/{epoch}'+'='*30)
        t_loss, t_macro_f1, t_acc, t_uar = train(train_loader)
        if (e == epoch-1):
            v_loss, v_macro_f1, v_acc, v_uar = evaluate(valid_loader, last=True,log_view=True)
        else:
            v_loss, v_macro_f1, v_acc, v_uar = evaluate(valid_loader,log_view=True)

        metrics = {
            "train/loss": t_loss,
            "train/macro_f1": t_macro_f1,
            "train/accuracy": t_acc,
            "train/uar": t_uar,
            "val/loss": v_loss,
            "val/macro_f1": v_macro_f1,
            "val/accuracy": v_acc,
            "val/uar": v_uar,
        }
        wandb.log(metrics)
        # train_losses.append(t_loss)
        # val_losses.append(v_loss)
        # train_macro_f1.append(t_macro_f1)
        # val_macro_f1.append(v_macro_f1)
        # train_acc.append(t_acc)
        # val_acc.append(v_acc)
    
    # Generate and save training report
    # write_training_report(file_uni_name, batch, epoch, leanring_rate, total_params, train_losses, val_losses, train_macro_f1, val_macro_f1,
    #                      train_acc, val_acc, start_time)
    wandb.finish()
    
    # save model parameters
    torch.save(model.state_dict(), os.path.join(config['PATH_TO_CKPT'],f'{file_uni_name}_model.pth'))

