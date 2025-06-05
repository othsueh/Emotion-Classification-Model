from .base import *

class CombineCorpus(BaseDataset):
    def __init__(self,dataset_dir):
        super().__init__(dataset_dir)
        # Need to change when using different dataset
        # [
        # Full Version
        # self.emotions = ['frustrated',
        #     'neutral',
        #     'happy',
        #     'angry',
        #     'surprise',
        #     'sad',
        #     'fear',
        #     'disgust',
        #     'contempt']
        # 6emo version
        self.emotions = [
            'neutral',
            'happy',
            'sad',
            'angry',
            'surprise',
            'contempt']

        # 6emo 10% version
        # self.sample_per_class = [3213, 1864, 602, 500, 269, 229]
        # self.train_counts = 6677
        # self.validation_counts = 1651
        # 6emo Full version
        self.sample_per_class = [38440, 22781, 7607, 7869, 3734, 3798]
        self.train_counts = 67725
        self.validation_counts = 16504
        # 6emo Full ORG
        # self.sample_per_class = [38592, 22951, 8021, 7794, 3920, 3846]
        # self.train_counts = 68620
        # self.validation_counts = 16504

        # 10% version
        # self.sample_per_class = [8, 3213, 1864, 500, 269, 602, 84, 118, 229]
        # self.train_counts = 6887
        # self.validation_counts = 1768
        # Full version
        # self.sample_per_class = [76, 32131, 18645, 5003, 2687, 6016, 844, 1184, 2287]
        # self.train_counts = 68873
        # self.validation_counts = 17670
        # ]
        self.test_counts = 45462

        
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

