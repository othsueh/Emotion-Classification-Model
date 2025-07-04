import tomli
import os
import io
import time
import torch
import wandb
import random
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from display import format_time
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')

with open("config.toml", "rb") as f:
    config = tomli.load(f)

with open("experiments_config.toml", "rb") as f:
    experiments_config = tomli.load(f)


def get_feature_dir(corpus,model):
    path = os.path.join(config[corpus]["PATH_TO_FEATURE"], model)
    assert os.path.exists(path), f"Feature directory {path} does not exist"
    return path

def log_view_table(dataset, audios, sr, predicted, labels, arousal, valence, true_arousalAndValence ,probs):
    columns = ["Audio", "Predict", "Target", "Arousal", "Valence", "True Arousal", "True Valence"] + dataset.emotions
    table = wandb.Table(columns=columns)
    for audio, pred, tar, aro, val, trueav, prob in zip(audios,predicted,labels,arousal, valence, true_arousalAndValence ,probs):
        table.add_data(wandb.Audio(audio, sample_rate=sr),dataset.index_to_emotion(pred),dataset.index_to_emotion(tar),aro, val ,trueav[0], trueav[1],*prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

def mixup_data(text, audio, label, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = text.shape[0]
    index = torch.randperm(batch_size)

    mixed_text = lam * text + (1 - lam) * text[index,:]
    mixed_audio = lam * audio + (1 - lam) * audio[index,:]
    label_a, label_b = label, label[index]
    return mixed_text, mixed_audio, label_a, label_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def write_training_report(*args, **kwargs):
    """
    Write a training report with model performance metrics and visualizations
    
    Args:
        *args: Variable length argument list:
            - file_uni_name (str): Name of unify file
            - batch (int): Batch size
            - epoch (int): Number of training epochs
            - learning_rate (float): Learning rate
            - total_params (int): Total number of model parameters
            - train_losses (list): List of training losses per epoch
            - val_losses (list): List of validation losses per epoch  
            - train_macro_f1 (list): List of training macro F1 scores per epoch
            - val_macro_f1 (list): List of validation macro F1 scores per epoch
            - train_acc (list): List of training accuracies per epoch
            - val_acc (list): List of validation accuracies per epoch
            - start_time (float): Training start time in seconds
            
        **kwargs: Additional keyword arguments
    """
    file_uni_name, batch, epoch, learning_rate, total_params, train_losses, val_losses, train_macro_f1, val_macro_f1, train_acc, val_acc, start_time = args
    total_time = time.time() - start_time
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(train_losses) + 1))
    plt.savefig(os.path.join(config['PATH_TO_PERFORMACE'], f'{file_uni_name}_loss.png'))
    plt.close()
    
    # Plot macro F1 and accuracy scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_macro_f1) + 1), train_macro_f1, label='Training Macro F1', linestyle='-')
    plt.plot(range(1, len(val_macro_f1) + 1), val_macro_f1, label='Validation Macro F1', linestyle='-')
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy', linestyle='--')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Training and Validation Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(train_macro_f1) + 1))
    plt.savefig(os.path.join(config['PATH_TO_PERFORMACE'], f'{file_uni_name}_metrics.png'))
    plt.close()
    
    # Create report markdown
    report = f"""
# Training Report

## Model Information
- Total Parameters: {total_params:,}
- Total Training Time: {format_time(total_time)}
- Batch Size: {batch}
- Learning Rate: {learning_rate}
- Number of Epochs: {epoch}

## Performance Metrics
### Training
- Macro F1 Score: {train_macro_f1[-1]:.3f}
- Accuracy: {train_acc[-1]:.3f}

### Validation  
- Macro F1 Score: {val_macro_f1[-1]:.3f}
- Accuracy: {val_acc[-1]:.3f}

## Visualizations
### Loss Curves
![Loss Plot]({os.path.join(".", f'{file_uni_name}_loss.png')})

### Performance Metrics
![Metrics Plot]({os.path.join(".", f'{file_uni_name}_metrics.png')})

### Confusion Matrix
![Confusion Matrix]({os.path.join(".", f'{file_uni_name}_confu.png')})
        """
    # Write report to file
    report_path = os.path.join(config['PATH_TO_PERFORMACE'], f'{file_uni_name}_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Training report saved to {report_path}")


# ================ Archive Space ====================
"""

def onehot_MSPPODCAST(emotion):
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]
    one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}
    return one_hot_dict[emotion]

def indexToEmotion(index):
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]
    emotion_code = emotion_codes[index]
    return emotion_code

def indexToFullEmotion(index):
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    mapping = {e: emotions[i] for i, e in enumerate(emotion_codes)}
    emotion_code = emotion_codes[index]
    return mapping[emotion_code]

class MSPDataset(IterableDataset):
    #Can create 2 kinds of dataset, one of original audio, another of feature generated by large model. 
    def __init__(self, df, audio_path=None, textFeature_path=None, audioFeature_path=None, use_feature=False, transform=default_transform,seed=42):
        super(MSPDataset, self).__init__()
        
        if use_feature:
            assert textFeature_path is not None, "textFeature_path cannot be None"
            self.textFeature_path = textFeature_path
            assert audioFeature_path is not None, "audioFeature_path cannot be None"
            self.audioFeature_path = audioFeature_path
        else:
            assert audio_path is not None, "audio_path cannot be None when use_feature is False"

        self.df = df
        self.audio_path = audio_path
        self.transform = transform
        self.seed = seed
        self.use_feature = use_feature

        # Total length for worker distribution
        self.total_samples = len(self.df)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        df = self.df
        if worker_info:
            per_worker = self.total_samples // worker_info.num_workers
            worker_id = worker_info.id

            # Set the random shuffle
            rng = random.Random(self.seed + worker_id)
            indices = list(range(len(df)))
            rng.shuffle(indices)

            # Set the start and end indices for the worker
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else None

        # Handle case when worker_info is None
        if not worker_info:
            indices = list(range(len(df)))
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            start_idx = 0
            end_idx = None

        for idx in indices[start_idx:end_idx]:
            row = df.iloc[idx]
            try:
                data, label = self._load_data(row)
                if self.transform:
                    data,label = self.transform(data,label,self.use_feature)
                yield data,label
            except Exception as e:
                print(f"Error loading file {row['FileName']}: {e}")
                continue

    def _load_data(self, row):

        name = row["FileName"]
        if self.use_feature:
            text_file = os.path.join(self.textFeature_path, name + ".npy")
            audio_file = os.path.join(self.audioFeature_path, name + ".npy")
            
            if not os.path.exists(text_file):
                raise FileNotFoundError(f"Text feature file not found: {text_file}")
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio feature file not found: {audio_file}")

            text_features = np.load(text_file)
            audio_features = np.load(audio_file)
            
            data = {
                "text": text_features,
                "audio": audio_features,
            }
        else:
            audio_file = os.path.join(self.audio_path,name + ".wav")
            
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            sample, sr = sf.read(audio_file)

            data = {
                "sample": sample,
                "sr": sr,
            }
            
        category = onehot_MSPPODCAST(row["EmoClass"])
        avd = [float(row['EmoAct']), float(row['EmoVal']), float(row['EmoDom'])]  # Convert to float
        
        label = {
            "category": category,
            "avd": avd
        }
        return data, label

class MSPTestset(Dataset):
    def __init__(self, df, text_path, audio_path, transform=default_transform):
        self.df = df
        self.text_path = text_path
        self.audio_path = audio_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            data = self._load_data(row)
            if self.transform:
                data = self.transform(data, None)
            return data
        except Exception as e:
            print(f"Error loading file {row['FileName']}: {e}")
            return None

    def _load_data(self, row):
        name = row["FileName"]
        text_file = os.path.join(self.text_path, name + '.npy')
        audio_file = os.path.join(self.audio_path, name + '.npy')
        
        # Check if files exist before loading
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text feature file not found: {text_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio feature file not found: {audio_file}")

        text_features = np.load(text_file)
        audio_features = np.load(audio_file)

        data = {
            "name": name,
            "text": text_features,
            "audio": audio_features
        }
        return data
"""
