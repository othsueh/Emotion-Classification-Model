import io
import glob
import json
import webdataset as wds
from utils import *


class BaseDataset:
    def __init__(self,dataset_dir):
        self.dataset_dir = dataset_dir
        self.genders = [
            'Male',
            'Female',
            'Unknown']
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
        
        return [self.emotions[idx] for idx in indices]
    
    def emotion_to_onehot(self, meta_data):
        """
        Convert dominant emotion to one-hot encoded vector.
        
        Args:
            meta_data (dict): Dictionary containing dominant emotion
            
        Returns:
            torch.Tensor: One-hot encoded vector of shape (8,)
        """

        # Create a zero tensor of length 8 (number of emotions)
        one_hot = torch.zeros(len(self.emotions), dtype=torch.float32)
        
        # get the index of dominant emotion
        emotion = meta_data['dominant_emotion']
        
        # Find the index of the emotion in our list
        try:
            emotion_idx = self.emotions.index(emotion)
            # Set 1 at the index of the emotion
            one_hot[emotion_idx] = 1.0
        except ValueError:
            # Handle case where emotion is not in the list
            print(f"Warning: Emotion '{emotion}' not found in emotion list")
            
        return one_hot, emotion
    
    def batch_index_to_gender(self, indices):
        """
        Convert batch of indices to gender labels.
        
        Args:
            indices (torch.Tensor or list): Batch of indices
            
        Returns:
            list: List of gender labels
        """
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        
        return [self.genders[idx] for idx in indices]
    def gender_to_onehot(self, meta_data):
        """
        Convert gender to one-hot encoded vector.
        
        Args:
            meta_data (dict): Dictionary containing gender
            
        Returns:
            torch.Tensor: One-hot encoded vector of shape (8,)
        """

        # Create a zero tensor of length 3 (number of genders) with dtype float32
        one_hot = torch.zeros(3, dtype=torch.float32)
        
        # get the index of gender
        gender = meta_data['gender']
        
        # Find the index of the emotion in our list
        try:
            gender_idx = self.genders.index(gender)
            # Set 1 at the index of the emotion
            one_hot[gender_idx] = 1.0
        except ValueError:
            # Handle case where emotion is not in the list
            print(f"Warning: Gender '{gender}' not found in emotion list")
            
        return one_hot, gender


    def get_collate_fn(self):
        """Return the collate function for this dataset type"""
        return collate_fn
    
    def create_dataloader(self, split='train', batch_size=32, num_workers=4, shuffle=True):
        # Pattern to match all shards for the split
        if split == 'test': # For project use
            pattern = f"{self.dataset_dir}/{split}-samples-*.tar"
        else:
            pattern = f"{self.dataset_dir}/{split}/{split}-samples-*.tar"
        
        # Check if the pattern exists
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        # Use the actual matching files instead of the pattern
        dataset = (
            wds.WebDataset(matching_files, shardshuffle=False, ) # Don't know what's shardsshuffle
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
