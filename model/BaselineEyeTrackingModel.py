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
        # Sử dụng GRU hai chiều (BiGRU) với hidden_size = 128 -> output có kích thước 256
        self.bigru = nn.GRU(input_size=72160, hidden_size=128, num_layers=1, 
                            bidirectional=True, batch_first=True)
        # Self-Attention với input_dim = 256 (tương ứng với output của BiGRU)
        self.self_attention = SelfAttention(input_dim=256)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        # x có shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        print(f"\nnData Shape (batch_size, seq_len, channels, height, width) -> {x.shape}")
        x = x.view(batch_size * seq_len, channels, height, width)
        # Đảo vị trí chiều height và width nếu cần
        x = x.permute(0, 1, 3, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.bigru(x)  # output có shape (batch_size, seq_len, 256)
        x = self.self_attention(x)  # Áp dụng Self-Attention
        x = self.fc(x)  # output cuối có shape (batch_size, seq_len, 2)
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
    
class MambaModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MambaModule, self).__init__()
        # Hai nhánh xử lý độc lập
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )
        # Hợp nhất kết quả từ hai nhánh
        self.fuse = nn.Linear(out_features * 2, out_features)
        
    def forward(self, x):
        # x: (N, in_features)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # Nối kết quả từ hai nhánh theo chiều cuối cùng
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
        super(EfficientNetB0_BiGRU_SelfAttention, self).__init__()
        self.args = args
        # Backbone trích xuất đặc trưng từ từng khung hình
        self.backbone = EfficientNetBackbone(feature_dim=feature_dim, pretrained=True)
        # Module Mamba để tinh chỉnh đặc trưng ngay sau backbone
        self.mamba = MambaModule(in_features=feature_dim, out_features=feature_dim)
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

