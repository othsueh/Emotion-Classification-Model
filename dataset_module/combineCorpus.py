from .base import *

class CombineCorpus(BaseDataset):
    def __init__(self,dataset_dir):
        super().__init__(dataset_dir)
        self.emotions = ['frustrated',
            'neutral',
            'happy',
            'angry',
            'surprise',
            'sad',
            'fear',
            'disgust',
            'contempt']

        self.sample_per_class = [454, 38393, 22712, 7679, 3448, 7402, 1143, 1673, 3639]
        self.train_counts = 68873
        self.validation_counts = 17670
        self.test_counts = 45462

    def emotion_to_onehot(self, meta_data):
        """
        Convert dominant emotion to one-hot encoded vector.
        
        Args:
            meta_data (dict): Dictionary containing dominant emotion
            
        Returns:
            torch.Tensor: One-hot encoded vector of shape (8,)
        """

        # Create a zero tensor of length 8 (number of emotions)
        one_hot = torch.zeros(len(self.emotions))
        
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
        
    def get_collate_fn(self):
        """Custom collate function for this specific dataset type"""
        def sample_collate_fn(batch):
            # Extract all items
            audio_tensors = [item['audio'] for item in batch]
            
            categories = []
            avs = []
            padded_audio = []
            
            # Find max length for padding
            max_length = max([tensor.shape[-1] for tensor in audio_tensors])

            for item in batch:
                tensor = item['audio']
                meta_data = item['json']

                # Pad audio
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
                
                # Get Label
                one_hot, emotion_label = self.emotion_to_onehot(meta_data)
                categories.append(one_hot)
                av = torch.tensor([meta_data['EmoAct'], meta_data['EmoVal']], dtype=torch.float32)
                avs.append(av)
            
            return {
                'audio': torch.stack(padded_audio).float(),
                'category': torch.stack(categories).float(),
                'av': torch.stack(avs),
            }
        
        return sample_collate_fn

