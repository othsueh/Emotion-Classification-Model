from .base import *

class MSPPodcast(BaseDataset):
    def __init__(self,dataset_dir):
        super().__init__(dataset_dir)
        self.emotions = ['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy']
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
        """Custom collate function for this specific dataset type"""
        def sample_collate_fn(batch):
            # Extract all items
            audio_tensors = [item['audio'] for item in batch]
            texts = [item['text'] for item in batch]
            
            categories = []
            avds = []
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
                avd = torch.tensor([meta_data['EmoAct'], meta_data['EmoVal'], meta_data['EmoDom']], dtype=torch.float32)
                avds.append(avd)
            
            return {
                'audio': torch.stack(padded_audio).float(),
                'text': texts,
                'category': torch.stack(categories).float(),
                'avd': torch.stack(avds),
            }
        
        return sample_collate_fn

