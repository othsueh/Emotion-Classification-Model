"""
This is the v2 version of my upstream_finetune model. The v1 version is in commit dce83ab.
The v2 architecture's idea is inspired by the approach described in https://arxiv.org/abs/2210.16642.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor, AutoModel
from safetensors.torch import load_file

class UpstreamFinetuneConfig(PretrainedConfig):
    model_type = "wav2vec2-emodualhead"
    def __init__(
        self,
        origin_upstream_url = "facebook/wav2vec2-base-960h",
        upstream_model="wav2vec2-base-960h",  # Reference to base model
        finetune_layers = 0 , # Prevent overhead gpu usage
        hidden_dim = 64, 
        dropout=0.2, 
        num_layers=2, 
        classifier_output_dim=8,
        regressor_output_dim=2,
        **kwargs
    ):
        self.origin_upstream_url = origin_upstream_url
        self.upstream_model = upstream_model
        self.dropout = dropout
        self.finetune_layers = finetune_layers
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.classifier_output_dim = classifier_output_dim
        self.regressor_output_dim = regressor_output_dim
        super().__init__(**kwargs)

class RegressionHead(nn.Module):
    def __init__(self,first_dim,hidden_dim,dropout, num_layers, min_score = 0.0, max_score = 1.0):
        super().__init__()
        self.min_score = min_score
        self.max_score = max_score

        layers = []
        input_dim = first_dim
        for i in range(num_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            
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
        output = torch.sigmoid(output) * (self.max_score - self.min_score) + self.min_score
        return output


class ClassificationHead(nn.Module):
    def __init__(self, first_dim, hidden_dim, dropout, num_layers, num_labels):
        super().__init__()
        self.hidden_layers = nn.Sequential(*[
            layer for i in range(num_layers)
            for layer in (nn.Linear(first_dim if i == 0 else hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        ])
        self.out_proj = nn.Linear(hidden_dim, num_labels)
        self.embedding_dim = hidden_dim

    def forward(self, x, return_embedding=False):
        embedding = self.hidden_layers(x)
        output = self.out_proj(embedding)
        return (output, embedding) if return_embedding else output

class HierarchicalDCRegressionHead(nn.Module):
    def __init__(self, classifier_embed_dim, cont_embed_dim, dropout, min_score=0.0, max_score=1.0):
        super().__init__()
        self.min_score = min_score
        self.max_score = max_score
        self.fusion_layer = nn.Sequential(
            nn.Linear(classifier_embed_dim + cont_embed_dim, cont_embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(cont_embed_dim, 2)
        )

    def forward(self, ed, ec):
        x = torch.cat([ed, ec], dim=-1)
        out = self.fusion_layer(x)
        return torch.sigmoid(out) * (self.max_score - self.min_score) + self.min_score


class UpstreamFinetune(PreTrainedModel):
    config_class = UpstreamFinetuneConfig
    def __init__(self, config, pretrained_path = None,device = None):
        super().__init__(config)
        if pretrained_path is None:
            upstream_path = config.origin_upstream_url
        else:
            upstream_path = os.path.join(pretrained_path, config.upstream_model)
        self.feature_extractor = AutoProcessor.from_pretrained(upstream_path,use_fast=False)
        self.upstream = AutoModel.from_pretrained(upstream_path)
        self.finetune_layers = config.finetune_layers
        
        # Explicitly initialize the masked_spec_embed parameter if it's causing issues
        if hasattr(self.upstream, 'masked_spec_embed'):
            self.upstream.masked_spec_embed = nn.Parameter(torch.zeros(self.upstream.config.hidden_size))
    
        for param in self.upstream.parameters():
            param.requires_grad = False
            
        for i in range(1, self.finetune_layers + 1):
            for param in self.upstream.encoder.layers[-i].parameters():
                param.requires_grad = True
                
        input_dim = self.upstream.config.hidden_size
        self.classifier = ClassificationHead(input_dim, config.hidden_dim, config.dropout, config.num_layers, config.classifier_output_dim)
        self.cont_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout)
        )
        self.regressor = HierarchicalDCRegressionHead(
            classifier_embed_dim=config.hidden_dim,
            cont_embed_dim=config.hidden_dim,
            dropout=config.dropout
        )
        self.to(device)
        
    def forward(self, x, sr):
        with torch.no_grad():   
            # Extract features from upstream model
            features = self.feature_extractor(x, sampling_rate=sr, return_tensors='pt', padding=True).input_values
            features = features.squeeze(0).squeeze(1)
            features = features.cuda()
        
            if torch.isnan(features).any():
                print("Warning: NaN detected in features")
                features = torch.nan_to_num(features, nan=0.0)
        
        # Process through upstream model
        outputs = self.upstream(features)
        hidden_states = outputs.last_hidden_state
        
        # For using multiple hidden states
        # upstream_hidden_state = self.upstream(features,output_hidden_states=True).hidden_states
        # upstream_hidden_state = torch.stack(upstream_hidden_state[-1:])
        # upstream_hidden_state = torch.mean(upstream_hidden_state, dim=0)
        
        # DEBUG field
        if torch.isnan(hidden_states).any():
            print("Warning: NaN detected in hidden state")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        
        # Global average pooling over the sequence length
        pooled_features = torch.mean(hidden_states, dim=1)
        
        # DEBUG field
        if torch.isnan(pooled_features).any():
            print("Warning: NaN detected in pooled features")
        
        # Pass through classifier
        # Get discrete output and embedding
        category, ed = self.classifier(pooled_features, return_embedding=True)

        # Get continuous embedding
        ec = self.cont_proj(pooled_features)

        # Use ED and EC to predict continuous values
        dim = self.regressor(ed, ec)

        return category, dim

    @classmethod
    def from_pretrained(cls, model_path, pretrained_model_name_or_path = None, *model_args, **kwargs):
        # Extract config and device from kwargs if provided
        device = kwargs.pop('device', None)
        pretrained_path = kwargs.pop('pretrained_path', None)
        
        # Load the configuration
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(model_path, **kwargs)
        
        # Create model instance with the config
        model = cls(config=config, pretrained_path=pretrained_model_name_or_path, device=device, *model_args, **kwargs)
        
        model_bin_path = os.path.join(model_path, "pytorch_model.bin")
        model_safetensors_path = os.path.join(model_path, "model.safetensors")

        if os.path.exists(model_safetensors_path):
            print(f"Loading model weights from {model_safetensors_path}...")
            state_dict = load_file(model_safetensors_path)
            model.load_state_dict(state_dict)
        elif os.path.exists(model_bin_path):
            print(f"Loading model weights from {model_bin_path}...")
            state_dict = torch.load(model_bin_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model weights found at {model_path}. Expected either 'pytorch_model.bin' or 'model.safetensors'")
        
        # Set model to eval mode by default
        model.eval()
        
        return model
        
        
