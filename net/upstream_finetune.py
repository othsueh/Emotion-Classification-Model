import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

class RegressionHead(nn.Module):
    def __init__(self,first_dim,hidden_dim,dropout, num_layers):
        super().__init__()

        layers = []
        input_dim = first_dim
        for i in range(num_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            # Initialize weights using Xavier/Glorot initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        # Final classification layer
        final_layer = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.regressor = nn.Sequential(*layers)

    def forward(self,x):
        output = self.regressor(x)
        return output


class ClassificationHead(nn.Module):
    def __init__(self,first_dim,hidden_dim,dropout, num_layers, num_labels):
        super().__init__()

        layers = []
        input_dim = first_dim
        for i in range(num_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            # Initialize weights using Xavier/Glorot initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        # Final classification layer
        final_layer = nn.Linear(hidden_dim, num_labels)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.classifier = nn.Sequential(*layers)

    def forward(self,x):
        output = self.classifier(x)
        return output

class Upstream_finetune_simple(nn.Module):
    def __init__(self, upstream_name, finetune_layers = 2 , hidden_dim = 64, dropout=0.2, num_layers=2, num_labels=8, device='cuda'):
        super().__init__()
        
        self.feature_extractor = AutoProcessor.from_pretrained("/datas/store163/othsueh/FeatureExtraction/models/wav2vec2-base-960h",use_fast=False)
        self.upstream = AutoModel.from_pretrained("/datas/store163/othsueh/FeatureExtraction/models/wav2vec2-base-960h")
        self.finetune_layers = finetune_layers

        for param in self.upstream.parameters():
            param.requires_grad = False
            
        for i in range(1, self.finetune_layers + 1):
            for param in self.upstream.encoder.layers[-i].parameters():
                param.requires_grad = True
                
        input_dim = self.upstream.config.hidden_size
        self.classifier = ClassificationHead(input_dim,hidden_dim,dropout,num_layers,num_labels)
        self.regressor = RegressionHead(input_dim,hidden_dim,dropout,num_layers)

        self.to(device)
        
    def forward(self, x, sr):
        # Extract features from upstream model
        features = self.feature_extractor(x,sampling_rate=sr,return_tensors='pt',padding=True).input_values
        features = features.squeeze(0)
        features = features.squeeze(1)
        features = features.cuda()
        
        if torch.isnan(features).any():
            print("Warning: NaN detected in features")
            features = torch.nan_to_num(features, nan=0.0)
        
        # For using multiple hidden states
        # upstream_hidden_state = self.upstream(features,output_hidden_states=True).hidden_states
        # upstream_hidden_state = torch.stack(upstream_hidden_state[-1:])
        # upstream_hidden_state = torch.mean(upstream_hidden_state, dim=0)
        
        upstream_hidden_state = self.upstream(features).last_hidden_state
        
        # DEBUG field
        if torch.isnan(upstream_hidden_state).any():
            print("Warning: NaN detected in hidden state")
            upstream_hidden_state = torch.nan_to_num(upstream_hidden_state, nan=0.0)
        
        # Global average pooling over the sequence length
        pooled_features = torch.mean(upstream_hidden_state, dim=1)
        
        # DEBUG field
        if torch.isnan(pooled_features).any():
            print("Warning: NaN detected in pooled features")
        
        # Pass through classifier
        category = self.classifier(pooled_features)
        dim = self.regressor(pooled_features)
        
        # DEBUG field
        if torch.isnan(category).any():
            print("Warning: NaN detected in classifier output")

        return category, dim
        
        
