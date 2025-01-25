import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleModel, self).__init__()
        input_dim = args[0]
        drop = kwargs.get('dropout', 0.1)
        hidden_dim = 1024
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1024)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=drop,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # MLP layers
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.dropout1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(512, 8)
        
    def forward(self, x):
        # Apply pooling and reshape for transformer
        x = self.pool(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer with residual connection
        x = self.norm1(x)
        x_t = self.transformer(x)
        x = x + x_t  # Residual connection
        x = self.norm2(x)
        
        # Squeeze sequence dimension
        x = x.squeeze(1)
        
        # MLP layers with GELU activation
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.fc2(x)
        # Output layer with softmax
        # x = F.softmax(self.fc2(x), dim=-1)
        return x