import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

class Upstream_finetune_simple(nn.Module):
    def __init__(self, upstream_name, finetune_layers = 2 , hidden_dim = 64, dropout=0.2, num_layers=2, num_labels=8, device='cuda'):
        super().__init__()
        
        self.feature_extractor = AutoProcessor.from_pretrained("/workspace/Emotion_Recognition/models/wav2vec2-base-960h",use_fast=False)
        self.upstream = AutoModel.from_pretrained("/workspace/Emotion_Recognition/models/wav2vec2-base-960h")
        # self.upstream.config.ctc_zero_infinity = True
        self.finetune_layers = finetune_layers
        self.layer_norm = nn.LayerNorm(self.upstream.config.hidden_size)


        for param in self.upstream.parameters():
            param.requires_grad = False
            
        for i in range(1, self.finetune_layers + 1):
            for param in self.upstream.encoder.layers[-i].parameters():
                param.requires_grad = True
                
        # Downstream classifier layers
        layers = []
        input_dim = self.upstream.config.hidden_size
        
        # Create multiple layers based on num_layers argument
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        # Final classification layer
        layers.append(nn.Linear(hidden_dim, num_labels))
        
        self.classifier = nn.Sequential(*layers)

        self.to(device)
        
    def forward(self, x):
        # Extract features from upstream model
        features = self.feature_extractor(x,sampling_rate=16000,return_tensors='pt',padding=True).input_values
        features = features.squeeze(0)
        features = features.cuda()
        
        # Normalize input features
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        if torch.isnan(features).any():
            print("Warning: NaN detected in features")
            features = torch.nan_to_num(features, nan=0.0)
        
        # Add gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        upstream_hidden_state = self.upstream(features).last_hidden_state
        upstream_hidden_state = self.layer_norm(upstream_hidden_state)

        if torch.isnan(upstream_hidden_state).any():
            print("Warning: NaN detected in hidden state")
            upstream_hidden_state = torch.nan_to_num(upstream_hidden_state, nan=0.0)
        
        # Global average pooling over the sequence length
        pooled_features = torch.mean(upstream_hidden_state, dim=1)
        
        if torch.isnan(pooled_features).any():
            print("Warning: NaN detected in pooled features")
        
        # Pass through classifier
        output = self.classifier(pooled_features)
        
        if torch.isnan(output).any():
            print("Warning: NaN detected in output")

        return output
        
        
