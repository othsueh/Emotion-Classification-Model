import io
import glob
import json
import webdataset as wds
from utils import *


class BaseDataset:
    def __init__(self,dataset_dir):
        self.dataset_dir = dataset_dir
        pass
    
    def index_to_emotion(self, index):
        """
        Convert numerical index to emotion label.
        
        Args:
            index (int or torch.Tensor): Index of the emotion (0-7)
            
        Returns:
            str: Corresponding emotion label
            
        Raises:
            IndexError: If index is out of bounds
        """
        # Handle both tensor and integer inputs
        if torch.is_tensor(index):
            index = index.item()
        
        try:
            return self.emotions[index]
        except IndexError:
            raise IndexError(f"Index {index} is out of bounds. Must be between 0 and {len(self.emotions)-1}")

    def batch_index_to_emotion(self, indices):
        """
        Convert batch of indices to emotion labels.
        
        Args:
            indices (torch.Tensor or list): Batch of indices
            
        Returns:
            list: List of emotion labels
        """
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        
        return [self.index_to_emotion(idx) for idx in indices]
    
    def emotion_to_onehot(self, meta_data):
        """
        Convert emotion probabilities to one-hot encoded vector based on max probability.
        
        Args:
            meta_data (dict): Dictionary containing emotion probabilities
            
        Returns:
            torch.Tensor: One-hot encoded vector of shape (8,)
        """
        # Create a zero tensor of length 8 (number of emotions)
        one_hot = torch.zeros(len(self.emotions))
        
        # Find emotion with maximum probability
        max_emotion = None
        max_probability = 0
        for emo in self.emotions:
            if meta_data[emo] > max_probability:
                max_probability = meta_data[emo]
                max_emotion = emo
        
        # Set 1 at the index of the max emotion
        if max_emotion is not None:
            emotion_idx = self.emotions.index(max_emotion)
            one_hot[emotion_idx] = 1.0
            
        return one_hot, max_emotion
    def get_collate_fn(self):
        """Return the collate function for this dataset type"""
        return collate_fn
    
    def create_dataloader(self, split='train', batch_size=32, num_workers=4, shuffle=True):
        # Pattern to match all shards for the split
        pattern = f"{self.dataset_dir}/{split}/{split}-samples-*.tar"
        
        # Check if the pattern exists
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        # Use the actual matching files instead of the pattern
        dataset = (
            wds.WebDataset(matching_files, shardsshuffle=False, ) # Don't know what's shardsshuffle
            .shuffle(1000)
            .map_dict(
                audio=lambda x: torch.load(io.BytesIO(x), weights_only=True),
                json=lambda x: json.loads(x.decode('utf-8'))
            )
            .select(lambda x: x['audio'] is not None)
        )

        # Adjust number of workers based on available shards
        effective_workers = min(num_workers, max(1, len(matching_files)))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=effective_workers,
            shuffle=False,
            collate_fn=self.get_collate_fn()  # Use the collate function from the class
        )
        
        return loader
    
    def plot_confusion_matrix(self, predictions, true_labels, file_uni_name):
        """
        Plot confusion matrix for emotion predictions
        Args:
            predictions: list of predicted emotion labels
            true_labels: list of true emotion labels
        """
        
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
                    xticklabels=self.emotions,
                    yticklabels=self.emotions)
        plt.title('Emotion Prediction Confusion Matrix')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        plt.savefig(os.path.join(config['PATH_TO_PERFORMACE'],
                    f'{file_uni_name}_confu.png'))
        plt.close()
    
    def log_view_table(self, audios, sr, predicted, labels, arousal_valence, probs):
        columns = ["Audio", "Predict", "Target", "Arousal", "Valence"] + self.emotions
        table = wandb.Table(columns=columns)
        for audio, pred, tar, aro_val, prob in zip(audios,predicted,labels,arousal_valence, probs):
            table.add_data(wandb.Audio(audio,sample_rate=sr),indexToFullEmotion(pred),indexToFullEmotion(tar),aro_val[0],aro_val[1],*prob.numpy()) 
        wandb.log({"predictions_table":table}, commit=False)

# Sample collate function
def collate_fn(batch):
    # Extract all items
    audio_tensors = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    jsons = [item['json'] for item in batch]
    
    # Find max length for padding
    max_length = max([tensor.shape[-1] for tensor in audio_tensors])
    
    # Pad audio tensors to the same length
    padded_audio = []
    for tensor in audio_tensors:
        # Assuming tensor shape is [channels, length]
        channels = tensor.shape[0]
        current_length = tensor.shape[-1]
        
        if current_length < max_length:
            # Create padding
            padding = torch.zeros(channels, max_length - current_length, 
                                    dtype=tensor.dtype, device=tensor.device)
            # Concatenate padding
            padded = torch.cat([tensor, padding], dim=1)
            padded_audio.append(padded)
        else:
            padded_audio.append(tensor)
    
    # Stack into a batch
    audio_batch = torch.stack(padded_audio)
    
    return {
        'audio': audio_batch,
        'text': texts,
        'json': jsons
    }
