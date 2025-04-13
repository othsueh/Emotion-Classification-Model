import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBottleneck(nn.Module):
    def __init__(self, indim, innerdim):
        hidden_dim = 256
        outdim = hidden_dim
        drop = 0.3
        super(TransformerBottleneck, self).__init__()

        # Downsample layer for dimension matching
        self.downsample = None
        if indim != outdim:
            self.downsample = nn.Sequential(
                nn.Linear(indim, outdim),
                nn.BatchNorm1d(outdim)
            )

        self.lin1 = nn.Linear(indim, innerdim)
        self.bn1 = nn.BatchNorm1d(innerdim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_dim,
            nhead=4,
            dim_feedforward=innerdim,
            dropout=drop,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.lin2 = nn.Linear(innerdim,outdim)
        self.bn2 = nn.BatchNorm1d(outdim)
        self.gelu = nn.GELU()

    def forward(self,x):
        # Store residual and apply downsample if needed
        residual = x if self.downsample is None else self.downsample(x)

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.encoder(out)
        out = self.lin2(out)
        out = self.bn2(out)

        out += residual
        out = self.gelu(out)

        return out

class FusionLayer(nn.Module):
    def __init__(self,fusion_type,hidden_dim):

        super(FusionLayer, self).__init__()
        # 1. Simple fusion techniques
        self.fusion_type = fusion_type
        
        # 2. Gating mechanism
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. FiLM-based fusion
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        # 4. Bilinear fusion
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Optional: Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, audio_features, text_features):
         # 1. Simple fusion methods
        if self.fusion_type == 'concat':
            fused = torch.cat([audio_features, text_features], dim=-1)
        elif self.fusion_type == 'sum':
            fused = audio_features + text_features
        elif self.fusion_type == 'average':
            fused = (audio_features + text_features) / 2
            
        # 2. Gated fusion
        elif self.fusion_type == 'gated':
            audio_gate = self.audio_gate(audio_features)
            text_gate = self.text_gate(text_features)
            fused = audio_gate * audio_features + text_gate * text_features
            
        # 3. FiLM-based fusion
        elif self.fusion_type == 'film':
            combined = torch.cat([audio_features, text_features], dim=-1)
            film_params = self.film_generator(combined)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            fused = gamma * audio_features + beta
            
        # 4. Bilinear fusion
        elif self.fusion_type == 'bilinear':
            fused = self.bilinear(audio_features, text_features)
        
        # Optional: Final projection
        fused = self.output_projection(fused)
        
        return fused
 
class Conv2DBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2DBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if dimensions don't match
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ExpertLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_experts=1, returns_feat=False):
        super(ExpertLayer, self).__init__()
        self.returns_feat = returns_feat 
        # Create ModuleList of experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 8)
            ) for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, in_channels, H, W]
            expert_idx: Optional, specific expert to use (None means use all experts)
        Returns:
            If expert_idx is None: List of outputs from all experts
            If expert_idx is specified: Output from specific expert
        """
        # Return outputs from all experts
        outs = [expert(x) for expert in self.experts]
        logits = torch.stack(outs,dim=1)
        final_out = torch.stack(outs,dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out, 
                "logits": logits
            }
        else:
            return final_out
        
class MOE(nn.Module):
    def __init__(self, daudio, dtext,**kwargs):
        hidden_dim = 256
        num_heads = 4
        drop = kwargs.get('dropout', 0.2)
        fusion_type = kwargs.get('fusion_type', 'concat')
        self.num_experts = kwargs.get('num_experts',3)
        super(MOE, self).__init__()

        self.audio_block = TransformerBottleneck(daudio,hidden_dim)
        self.text_block = TransformerBottleneck(dtext,hidden_dim)
        
        # Cross attention layers
        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )
        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )
        
        # Layer norms for cross attention
        self.layer_norm_audio = nn.LayerNorm(hidden_dim)
        self.layer_norm_text = nn.LayerNorm(hidden_dim)

        self.fusion_layer = FusionLayer(fusion_type,hidden_dim)
        
        self.expert_layer = ExpertLayer(
            in_channels=hidden_dim,
            hidden_dim=128,
            num_experts=self.num_experts,
            returns_feat=True
        )
        self.dropout = nn.Dropout(drop)

    def forward(self,audio_input, text_input):
        audio_features = self.audio_block(audio_input)
        text_features = self.text_block(text_input)
        
        #! Add dimension checks(DEBUG)
        assert audio_features.size(-1) == 256, f"Audio features dim {audio_features.size(-1)} != 256"
        assert text_features.size(-1) == 256, f"Text features dim {text_features.size(-1)} != 256"       

        # Cross attention: Audio attending to Text
        audio_attended, _ = self.audio_to_text_attention(
            query=audio_features,
            key=text_features,
            value=text_features
        )
        audio_features = audio_features + self.dropout(audio_attended)
        audio_features = self.layer_norm_audio(audio_features)
        
        # Cross attention: Text attending to Audio
        text_attended, _ = self.text_to_audio_attention(
            query=text_features,
            key=audio_features,
            value=audio_features
        )
        text_features = text_features + self.dropout(text_attended)
        text_features = self.layer_norm_text(text_features)

        fuse_features = self.fusion_layer(audio_features,text_features)

        expert_outputs = self.expert_layer(fuse_features)

        return expert_outputs


