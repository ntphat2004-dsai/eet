import torch
import torch.nn as nn
import torch.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CNN_GRU(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.gru = nn.GRU(input_size=36192, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)


    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        x= self.conv1(x)
        x= torch.relu(x)
        x= self.conv2(x)
        x= torch.relu(x)
        x= self.conv3(x)
        x= torch.relu(x)
        x= self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.gru(x)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x


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
        # Tính attention score: (batch_size, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(scores)
        out = torch.bmm(attn, V)
        return out

class CustomCNN_BiGRU_SelfAttention(nn.Module):
    """
    Mô hình dự đoán tâm đồng tử kết hợp custom CNN + BiGRU + SelfAttention.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # Applied (BiGRU) hidden_size = 128 -> output shape = 256
        self.bigru = nn.GRU(input_size=72160, hidden_size=128, num_layers=1, 
                            bidirectional=True, batch_first=True)
        # Self-Attention, input_dim = 256 ~ output BiGRU)
        self.self_attention = SelfAttention(input_dim=256)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        print(f"\nnData Shape (batch_size, seq_len, channels, height, width) -> {x.shape}")
        x = x.view(batch_size * seq_len, channels, height, width)
        
        x = x.permute(0, 1, 3, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.bigru(x)  # Output shape (batch_size, seq_len, 256)
        x = self.self_attention(x)  # Applied Self-Attention
        x = self.fc(x)  # Last output shape (batch_size, seq_len, 2)
        print(f"\nOutput Shape {x.shape}")
        return x

class EfficientNetBackbone(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        super(EfficientNetBackbone, self).__init__()
        # Load EfficientNetB0 với trọng số pretrained nếu có
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b0(weights=weights)
        
        # Lấy phần feature extractor của EfficientNet
        self.features = self.efficientnet.features
        
        # Nếu freeze=True, đóng băng tất cả tham số của backbone
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Adaptive Pooling & Flatten
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Projection Layer
        self.proj = nn.Linear(1280, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)  # Thêm BatchNorm
        self.act = nn.ReLU(inplace=True)       # Thêm Activation
    
    def forward(self, x):
        x = self.features(x)  # (N, 1280, H', W')
        x = self.avgpool(x)   # (N, 1280, 1, 1)
        x = self.flatten(x)   # (N, 1280)
        x = self.proj(x)      # (N, feature_dim)
        x = self.bn(x)        # Chuẩn hóa đặc trưng
        x = self.act(x)       # Áp dụng activation
        return x

class EfficientNetB0_BiGRU_SelfAttention(nn.Module):
    """
    Mô hình dự đoán tâm đồng tử sử dụng EfficientNetB0 làm Backbone cho phần trích xuất đặc trưng,
    sau đó dùng BiGRU và Self-Attention để xử lý chuỗi thời gian.
    
    Đầu vào: (batch_size, seq_len, channels, height, width)
    Output: (batch_size, seq_len, 2)
    """
    def __init__(self, args, feature_dim=256, gru_hidden_size=128):
        super(EfficientNetB0_BiGRU_SelfAttention, self).__init__()
        self.args = args
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        # GRU input size bằng feature_dim của backbone
        self.bigru = nn.GRU(input_size=feature_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        # Self-Attention với input_dim = gru_hidden_size * 2 (do GRU là bidirectional)
        self.self_attention = SelfAttention(input_dim=gru_hidden_size*2)
        # Lớp fully-connected cuối cùng để dự đoán tọa độ (2 giá trị)
        self.fc = nn.Linear(gru_hidden_size*2, 2)
    
    def forward(self, x):
        # x có shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        # Gộp batch và seq lại để xử lý từng frame qua EfficientNetB0
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)  # Output: (batch_size * seq_len, feature_dim)
        # Chuyển về lại shape: (batch_size, seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.bigru(x)  # Output: (batch_size, seq_len, gru_hidden_size*2)
        x = self.self_attention(x)
        x = self.fc(x)        # Output: (batch_size, seq_len, 2)
        return x
    
class Mamba(nn.Module):
    def __init__(self, in_features, out_features):
        super(Mamba, self).__init__()
        # Independent branches
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )
        # Concatenate two branches
        self.fuse = nn.Linear(out_features * 2, out_features)
        
    def forward(self, x):
        # x: (N, in_features)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # Get output
        out = torch.cat([out1, out2], dim=-1)
        out = self.fuse(out)
        return out

class EfficientNetB0_Mamba_BiGRU_SelfAttention(nn.Module):
    """
    Mô hình dự đoán vị trí đồng tử:
      - Đầu vào: (batch_size, seq_len, channels, height, width)
      - Đầu ra: (batch_size, seq_len, 2)
    """
    def __init__(self, args, feature_dim=256, gru_hidden_size=128):
        super(EfficientNetB0_Mamba_BiGRU_SelfAttention, self).__init__()
        self.args = args
        # Backbone trích xuất đặc trưng từ từng khung hình
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        # Module Mamba để tinh chỉnh đặc trưng ngay sau backbone
        self.mamba = Mamba(in_features=feature_dim, out_features=feature_dim)
        # GRU xử lý chuỗi thời gian, với input_size = feature_dim
        self.bigru = nn.GRU(input_size=feature_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        # Self-Attention với input_dim = gru_hidden_size*2 (do GRU là bidirectional)
        self.self_attention = SelfAttention(input_dim=gru_hidden_size*2)
        # Lớp Fully-Connected cuối cùng để dự đoán tọa độ (2 giá trị: x, y)
        self.fc = nn.Linear(gru_hidden_size*2, 2)
    
    def forward(self, x):
        # x có shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        # Gộp batch và seq lại để xử lý từng frame qua EfficientNetB0
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)  # Output: (batch_size * seq_len, feature_dim)
        # Áp dụng Mamba để làm giàu đặc trưng
        x = self.mamba(x)     # Vẫn giữ shape: (batch_size * seq_len, feature_dim)
        # Reshape về lại dạng chuỗi: (batch_size, seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)
        # Xử lý chuỗi qua BiGRU: output có shape (batch_size, seq_len, gru_hidden_size*2)
        x, _ = self.bigru(x)
        # Áp dụng Self-Attention
        x = self.self_attention(x)
        # Dự đoán tọa độ (x, y) cho mỗi khung hình
        x = self.fc(x)        # Output: (batch_size, seq_len, 2)
        return x

class MambaSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(MambaSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Parameters for the state-space model
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)

        # Non-linearity
        self.activation = nn.GELU()
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            u_t = x[:, t, :]  # (batch_size, input_dim)
            h = self.activation(self.A @ h.T + self.B @ u_t.T).T  # Update hidden state
            y_t = self.C @ h.T  # Output
            outputs.append(y_t.T)

        out = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        return self.ln(out)

class EfficientNetB0_Mamba_BiGRU_SelfAttention_V2(nn.Module):
    """
    Mô hình dự đoán tâm đồng tử sử dụng:
    - EfficientNetB0 để trích xuất đặc trưng hình ảnh
    - MambaSSM để xử lý chuỗi trước khi vào BiGRU
    - BiGRU để học quan hệ thời gian
    - Self-Attention để khuếch đại tín hiệu quan trọng
    - Fully Connected để dự đoán tọa độ (x, y)
    
    Đầu vào: (batch_size, seq_len, channels, height, width)
    Đầu ra: (batch_size, seq_len, 2)
    """
    def __init__(self, args, feature_dim=256, mamba_hidden_dim=256, seq_len=10, gru_hidden_size=128):
        super(EfficientNetB0_Mamba_BiGRU_SelfAttention_V2, self).__init__()
        self.args = args
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        
        # Mamba State Space Model
        self.mamba = MambaSSM(input_dim=feature_dim, hidden_dim=mamba_hidden_dim, seq_len=seq_len)
        
        # BiGRU
        self.bigru = nn.GRU(input_size=mamba_hidden_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        
        # Self-Attention
        self.self_attention = SelfAttention(input_dim=gru_hidden_size*2)
        
        # Fully Connected để dự đoán tọa độ (x, y)
        self.fc = nn.Linear(gru_hidden_size*2, 2)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch và seq lại để đưa qua EfficientNetB0
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        
        # Chuyển về lại dạng chuỗi (batch_size, seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)
        
        # Đi qua MambaSSM
        x = self.mamba(x)  # (batch_size, seq_len, mamba_hidden_dim)
        
        # Đi qua BiGRU
        x, _ = self.bigru(x)  # (batch_size, seq_len, gru_hidden_size*2)
        
        # Self-Attention
        x = self.self_attention(x)
        
        # Dự đoán tọa độ
        x = self.fc(x)  # (batch_size, seq_len, 2)
        return x


class EfficientNetB0_BiGRU_MambaSSM(nn.Module):
    """ 
    Mô hình dự đoán tâm đồng tử sử dụng:
    - EfficientNetB0 làm Backbone cho phần trích xuất đặc trưng
    - BiGRU để học quan hệ thời gian
    - MambaSSM để xử lý chuỗi thời gian có chọn lọc
    - Fully Connected để dự đoán tọa độ (x, y)
    """
    def __init__(self, args, feature_dim=256, mamba_hidden_dim=256, seq_len=10, gru_hidden_size=128):
        super(EfficientNetB0_BiGRU_MambaSSM, self).__init__()
        self.args = args
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        self.mamba = MambaSSM(input_dim=feature_dim, hidden_dim=mamba_hidden_dim, seq_len=seq_len)
        self.bigru = nn.GRU(input_size=mamba_hidden_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size*2, 2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.mamba(x)
        x, _ = self.bigru(x)
        x = self.fc(x)
        return x