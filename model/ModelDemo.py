import torch
import torch.nn as nn
import torch.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ==================================================== #
# ================== Self Attention ================== #
# ==================================================== #
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Compute attention scores: (batch_size, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(scores)
        out = torch.bmm(attn, V)  # Shape: 
        return out


# ==================================================== #
# ================== EfficientNetB0 ================== #
# ==================================================== #
class EfficientNetBackbone(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        super(EfficientNetBackbone, self).__init__()
        # Load EfficientNetB0 with pretrained weights if available
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b0(weights=weights)
        
        # Extract the feature extractor part of EfficientNet
        self.features = self.efficientnet.features
        
        # If freeze=True, freeze all parameters of the backbone
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Adaptive Pooling & Flatten
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Projection Layer
        self.proj = nn.Linear(1280, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)  # Add BatchNorm
        self.act = nn.ReLU(inplace=True)       # Add Activation
    
    def forward(self, x):
        x = self.features(x)  # (N, 1280, H', W')
        x = self.avgpool(x)   # (N, 1280, 1, 1)
        x = self.flatten(x)   # (N, 1280)
        x = self.proj(x)      # (N, feature_dim)
        x = self.bn(x)        # Normalize features
        x = self.act(x)       # Apply activation
        return x
    

# ==================================================== #
# ===================== Mamba SSM  =================== #
# ==================================================== #
class MambaSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        """
        x: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, hidden_dim)
        """
        super(MambaSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Initialize the parameters of the state-space model with the desired dimensions
        # Matrix A: (hidden_dim, hidden_dim)
        self.A = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        # Matrix B: (hidden_dim, input_dim)
        self.B = nn.Parameter(torch.empty(hidden_dim, input_dim))
        # Matrix C: To ensure the output has shape (hidden_dim) for each time step,
        # we set C to have shape (hidden_dim, hidden_dim)
        self.C = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        # Non-linearity
        self.activation = nn.GELU()
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # Apply weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with orthogonal initialization
        nn.init.orthogonal_(self.A)
        # Initialize B and C with Xavier uniform
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        # Initialize hidden state, shape: (batch_size, hidden_dim)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            u_t = x[:, t, :]  # (batch_size, input_dim)
            # Compute h: transpose to match matrix multiplication, then transpose back
            h = self.activation((self.A @ h.T) + (self.B @ u_t.T)).T  # (batch_size, hidden_dim)
            y_t = (self.C @ h.T).T  # (batch_size, hidden_dim)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        return self.ln(out)


# ==================================================== #
# ========= EfficientNetB0_BiGRU_MambaSSM ============ #
# ==================================================== #
class EfficientNetB0_BiGRU_MambaSSM(nn.Module):
    """ 
    A model for pupil center prediction using:
    - EfficientNetB0 as the Backbone for feature extraction
    - BiGRU to learn temporal relationships
    - MambaSSM to selectively process temporal sequences
    - Fully Connected to predict coordinates (x, y)
    """
    def __init__(self, args, feature_dim=256, mamba_hidden_dim=256, seq_len=10, gru_hidden_size=128):
        super(EfficientNetB0_BiGRU_MambaSSM, self).__init__()
        self.args = args
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        self.bigru = nn.GRU(input_size=feature_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.mamba = MambaSSM(input_dim=gru_hidden_size*2, hidden_dim=mamba_hidden_dim, seq_len=seq_len)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(mamba_hidden_dim, 2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)                 # (batch_size * seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        x, _ = self.bigru(x)                 # (batch_size, seq_len, gru_hidden_size*2)
        x = self.mamba(x)                    # (batch_size, seq_len, mamba_hidden_dim)
        x = self.dropout(x)                  # dropout for regularization
        x = self.fc(x)                       # (batch_size, seq_len, 2)
        return x

# ==================================================== #
# ========== EfficientNetBackbone_unfreeze =========== #
# ==================================================== #
class EfficientNetBackbone_unfreeze(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        super(EfficientNetBackbone_unfreeze, self).__init__()
        # Load EfficientNetB0 with pretrained weights if available
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b0(weights=weights)
        
        # Extract the feature extractor part of EfficientNet
        self.features = self.efficientnet.features
        
        # If freeze=True, freeze all parameters of the backbone
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Adaptive Pooling & Flatten
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Projection Layer and Residual Connection:
        # The "proj" layer transforms features from 1280 to feature_dim,
        # while "residual_conv" directly maps the flattened input (1280) to feature_dim
        self.residual_conv = nn.Linear(1280, feature_dim)
        self.proj = nn.Linear(1280, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)  # BatchNorm for output
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (N, channels, height, width)
        x = self.features(x)   # (N, 1280, H', W')
        x = self.avgpool(x)    # (N, 1280, 1, 1)
        x = self.flatten(x)    # (N, 1280)
        
        # Compute skip branch: map flattened input to feature_dim
        res = self.residual_conv(x)  # (N, feature_dim)
        # Compute main branch
        proj_out = self.proj(x)      # (N, feature_dim)
        
        # Add the two branches, then normalize and apply activation
        out = self.bn(proj_out + res)
        out = self.act(out)
        return out

    def unfreeze_layers(self, num_layers=1):
        """
        Unfreeze the last num_layers of the backbone (self.features is nn.Sequential).
        For example: if num_layers=2, the last 2 blocks will be unfrozen.
        """
        children = list(self.features.children())
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True


# ==================================================== #
# EfficientNetB0_unfreeze_BiGRU_AttentionConv_MambaSSM #
# ===================================================- #
class EfficientNetB0_unfreeze_BiGRU_AttentionConv_MambaSSM(nn.Module):
    def __init__(self, args, feature_dim=256, mamba_hidden_dim=256, seq_len=10, gru_hidden_size=128, num_attention_heads=4):
        super(EfficientNetB0_unfreeze_BiGRU_AttentionConv_MambaSSM, self).__init__()
        self.args = args
        
        # Backbone: feature extraction from images
        self.backbone = EfficientNetBackbone_unfreeze(feature_dim=feature_dim, pretrained=True)
        
        # Bidirectional GRU, takes input of size feature_dim
        self.bigru = nn.GRU(
            input_size=feature_dim, 
            hidden_size=gru_hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Multi-head Self-Attention:
        # embed_dim = gru_hidden_size * 2 because GRU is bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size*2, 
            num_heads=num_attention_heads, 
            batch_first=True
        )
        
        # 1D Convolutional block: extracts local features along the temporal axis
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=gru_hidden_size*2, out_channels=gru_hidden_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=gru_hidden_size*2, out_channels=gru_hidden_size*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # MambaSSM processes the temporal sequence
        self.mamba = MambaSSM(input_dim=gru_hidden_size*2, hidden_dim=mamba_hidden_dim, seq_len=seq_len)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(mamba_hidden_dim, 2)

    def forward(self, x):
        """
        x: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Backbone: process each frame
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # GRU: process the sequence
        gru_out, _ = self.bigru(x)  # (batch_size, seq_len, gru_hidden_size*2)
        
        # Attention: use GRU output as query, key, value
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        gru_attn = gru_out + attn_out  # Residual connection
        
        # Convolutional block: reshape tensor for Conv1d
        conv_in = gru_attn.transpose(1, 2)  # (batch_size, features, seq_len)
        conv_out = self.conv1d(conv_in)
        conv_out = conv_out.transpose(1, 2)  # (batch_size, seq_len, features)
        
        # Combine results from GRU+Attention and Conv1d (e.g., add them)
        combined = gru_attn + conv_out
        
        # Process through MambaSSM and predict output
        mamba_out = self.mamba(combined)  # (batch_size, seq_len, mamba_hidden_dim)
        mamba_out = self.dropout(mamba_out)
        output = self.fc(mamba_out)       # (batch_size, seq_len, 2)
        return output

    def unfreeze_backbone(self, num_layers=1):
        """
        Call this method after a few epochs to unfreeze the last num_layers of the backbone for fine-tuning.
        """
        self.backbone.unfreeze_layers(num_layers=num_layers)