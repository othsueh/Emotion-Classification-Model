import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()  # Changed from ReLU to GELU for better gradient flow
        
    def forward(self, x):
        # Self-attention with gradient checkpointing
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        attn_output, _ = self.self_attn(x, x, x)
        x = residual + self.dropout1(attn_output)
        
        # Feedforward
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = residual + self.dropout3(x)
        
        return x

class AdaptiveHyperModalityLayer(nn.Module):
    def __init__(self, d_text, d_audio, d_hidden):
        super().__init__()
        
        # Initialize projections with Xavier/Glorot initialization
        self.text_proj = nn.Linear(d_text, d_hidden)
        self.audio_proj = nn.Linear(d_audio, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_hidden)
        
        # Initialize normalizations
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.norm_out = nn.LayerNorm(d_hidden)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Scaling factor for attention
        self.d_k = d_hidden
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.audio_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, H_l, H_a):
        # Project and normalize
        Q_l = self.norm1(self.text_proj(H_l))
        K_a = self.norm2(self.audio_proj(H_a))
        V_a = self.audio_proj(H_a)  # Share weights with K_a projection
        
        # Compute scaled dot-product attention
        scores = torch.matmul(Q_l, K_a.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attention with gradient scaling
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)  # Add dropout to attention weights
        
        # Compute weighted features with residual connection
        H_hyper = torch.matmul(alpha, V_a)
        H_hyper = self.out_proj(H_hyper)
        H_hyper = self.norm_out(H_hyper + H_l)  # Add residual connection
        
        return H_hyper

class MultimodalFusionNetwork(nn.Module):
    def __init__(
        self,
        d_text=1024,
        d_audio=1280,
        d_hidden=512,
        n_classes=8,
        n_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        # Input projections with layer norm
        self.text_input_proj = nn.Sequential(
            nn.Linear(d_text, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout)
        )
        
        self.audio_input_proj = nn.Sequential(
            nn.Linear(d_audio, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout)
        )
        
        # Modality-specific transformers
        self.text_transformer = nn.ModuleList([
            TransformerLayer(d_hidden, nhead=8, dropout=dropout)
            for _ in range(2)
        ])
        
        self.audio_transformer = TransformerLayer(d_hidden, nhead=8, dropout=dropout)
        
        # Adaptive Hyper-modality Learning layers
        self.ahl_layers = nn.ModuleList([
            AdaptiveHyperModalityLayer(d_hidden, d_hidden, d_hidden)
            for _ in range(n_layers)
        ])
        
        # Cross-modality Fusion
        self.fusion_transformer = TransformerLayer(d_hidden, nhead=8, dropout=dropout)
        
        # Classification head with layer norm
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(d_hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_classes)
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize all linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_features, audio_features):
        # Add sequence dimension if needed
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
            audio_features = audio_features.unsqueeze(1)
        
        # Initial projections
        text_features = self.text_input_proj(text_features)
        audio_features = self.audio_input_proj(audio_features)
        
        # Process each modality
        for layer in self.text_transformer:
            text_features = layer(text_features)
        
        audio_features = self.audio_transformer(audio_features)
        
        # Progressive fusion
        H = text_features
        for ahl_layer in self.ahl_layers:
            H = ahl_layer(H, audio_features)
            if torch.isnan(H).any():  # Add gradient check
                raise RuntimeError("NaN detected in fusion output")
        
        # Final fusion
        fused_features = self.fusion_transformer(H)
        
        # Global pooling with attention
        weights = F.softmax(torch.sum(fused_features, dim=-1, keepdim=True), dim=1)
        pooled_features = torch.sum(fused_features * weights, dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits

    def check_gradients(self):
        # Utility method to check gradient flow
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm == 0:
                    print(f"Warning: Zero gradient for {name}")
                elif torch.isnan(param.grad).any():
                    print(f"Warning: NaN gradient for {name}")