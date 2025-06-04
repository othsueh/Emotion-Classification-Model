from .base import *

class IEMOCAP(BaseDataset):
    def __init__(self,dataset_dir):
        super().__init__(dataset_dir)
        # 6emo version (For project use)
        self.emotions = [
            'neutral',
            'happy',
            'sad',
            'angry',
            'surprise',
            'contempt']
        self.test_counts = 1371
        # self.emotions = ['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy']
        # self.sample_per_class = [6716,6400,16652,2992,1134,1419,2519,29144]
        # self.train_counts = 84030
        # self.validation_counts = 19815
        # self.test_counts = 45462

    def get_collate_fn(self):
        """Custom collate function for this specific dataset type"""
        def sample_collate_fn(batch):
            # Extract all items
            audio_tensors = [item['audio'] for item in batch]
            
            categories = []
            avs = []
            genders = []
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
                gone_hot, gender = self.gender_to_onehot(meta_data)
                genders.append(gone_hot)
                av = torch.tensor([meta_data['EmoAct'], meta_data['EmoVal']], dtype=torch.float32)
                avs.append(av)
            
            return {
                'audio': torch.stack(padded_audio).float(),
                'gender': torch.stack(genders),
                'category': torch.stack(categories),
                'av': torch.stack(avs),
            }
        
        return sample_collate_fn

