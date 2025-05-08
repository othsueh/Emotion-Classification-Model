from .base import *

class MSPPodcast(BaseDataset):
    def __init__(self,dataset_dir):
        super().__init__(dataset_dir)
        self.emotions = ['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy']
        self.train_counts = 84030
        self.validation_counts = 19815
        self.test_counts = 45462

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

