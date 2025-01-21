import tomli
import os
import time
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from display import format_time


with open("config.toml", "rb") as f:
    config = tomli.load(f)


def onehot_MSPPODCAST(emotion):
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]
    one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}
    return one_hot_dict[emotion]

def indexToEmotion(index):
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    mapping = {e: emotions[i] for i, e in enumerate(emotion_codes)}
    emotion_code = emotion_codes[index]
    return mapping[emotion_code]

def get_feature_dir(corpus,model):
    path = os.path.join(config[corpus]["PATH_TO_FEATURE"], model)
    assert os.path.exists(path), f"Feature directory {path} does not exist"
    return path

def default_transform(data,label):
    data["text"] = torch.tensor(data["text"], dtype=torch.float32)
    data["audio"] = torch.tensor(data["audio"], dtype=torch.float32)
    label["category"] = torch.tensor(label["category"], dtype=torch.float32)
    label["avd"] = torch.tensor(label["avd"], dtype=torch.float32)
    return data, label

def corpus_split(corpus):
    train_df = corpus[corpus["Split_Set"] == "Train"]
    val_df = corpus[corpus["Split_Set"] == "Development"]
    return train_df, val_df

class MSPDataset(IterableDataset):
    def __init__(self, df, text_path, audio_path, transform=default_transform,seed=42):
        super(MSPDataset).__init__()
        self.df = df
        self.text_path = text_path
        self.audio_path = audio_path
        self.transform = transform
        self.seed = seed

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
                data, label = self._load_data(row, self.text_path, self.audio_path)
                if self.transform:
                    data,label = self.transform(data,label)
                yield data,label
            except Exception as e:
                print(f"Error loading file {row['FileName']}: {e}")
                continue

    def _load_data(self, row, text_path, audio_path):
        name = row["FileName"]
        text_file = os.path.join(text_path, name + ".npy")
        audio_file = os.path.join(audio_path, name + ".npy")
        
        # Check if files exist before loading
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text feature file not found: {text_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio feature file not found: {audio_file}")
            
        text_features = np.load(text_file)
        audio_features = np.load(audio_file)
            
        category = onehot_MSPPODCAST(row["EmoClass"])
        avd = [float(row['EmoAct']), float(row['EmoVal']), float(row['EmoDom'])]  # Convert to float
        
        data = {
            "text": text_features,
            "audio": audio_features,
        }
        label = {
            "category": category,
            "avd": avd
        }
        return data, label

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

def plot_confusion_matrix(predictions, true_labels, file_uni_name):
    """
    Plot confusion matrix for emotion predictions
    Args:
        predictions: list of predicted emotion labels
        true_labels: list of true emotion labels
    """
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    
    # Create confusion matrix
    confusion_matrix = np.zeros((8, 8))
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true][pred] += 1
    # Normalize confusion matrix with handling for zero division
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    row_sums = np.where(row_sums == 0, 1e-10, row_sums)
    confusion_matrix = confusion_matrix / row_sums
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap=sns.cubehelix_palette(as_cmap=True),
                xticklabels=emotions,
                yticklabels=emotions)
    plt.title('Emotion Prediction Confusion Matrix')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.tight_layout()
    plt.savefig(os.path.join(config['PATH_TO_PERFORMACE'],
                f'{file_uni_name}_confu.png'))
    plt.close()

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