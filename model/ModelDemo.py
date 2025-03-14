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
        # Tính attention score: (batch_size, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(scores)
        out = torch.bmm(attn, V)
        return out


# ==================================================== #
# ================== EfficientNetB0 ================== #
# ==================================================== #
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

        # Khởi tạo các tham số của state-space model với kích thước mong muốn
        # Ma trận A: (hidden_dim, hidden_dim)
        self.A = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        # Ma trận B: (hidden_dim, input_dim)
        self.B = nn.Parameter(torch.empty(hidden_dim, input_dim))
        # Ma trận C: Để đảm bảo đầu ra có shape (hidden_dim) cho mỗi bước thời gian,
        # ta sẽ đặt C có shape (hidden_dim, hidden_dim)
        self.C = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        # Non-linearity
        self.activation = nn.GELU()
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # Áp dụng khởi tạo trọng số
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo A với orthogonal initialization
        nn.init.orthogonal_(self.A)
        # Khởi tạo B và C với Xavier uniform
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        # Khởi tạo hidden state, shape: (batch_size, hidden_dim)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            u_t = x[:, t, :]  # (batch_size, input_dim)
            # Tính toán h: chuyển vị để phù hợp với phép nhân ma trận, sau đó transpose lại
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
# ==================== Model Demo ==================== #
# ==================================================== #
class EfficientNetBackbone_unfreeze(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, freeze=True):
        super(EfficientNetBackbone_unfreeze, self).__init__()
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
        
        # Projection Layer và Residual Connection:
        # Lớp "proj" chuyển đổi đặc trưng từ 1280 về feature_dim,
        # trong khi "residual_conv" chuyển trực tiếp flattened input (1280) về feature_dim
        self.residual_conv = nn.Linear(1280, feature_dim)
        self.proj = nn.Linear(1280, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)  # BatchNorm cho đầu ra
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (N, channels, height, width)
        x = self.features(x)   # (N, 1280, H', W')
        x = self.avgpool(x)    # (N, 1280, 1, 1)
        x = self.flatten(x)    # (N, 1280)
        
        # Tính nhánh skip: chuyển flattened input về feature_dim
        res = self.residual_conv(x)  # (N, feature_dim)
        # Tính nhánh chính
        proj_out = self.proj(x)      # (N, feature_dim)
        
        # Cộng hai nhánh, sau đó chuẩn hóa và áp dụng activation
        out = self.bn(proj_out + res)
        out = self.act(out)
        return out

    def unfreeze_layers(self, num_layers=1):
        """
        Unfreeze num_layers cuối cùng của backbone (self.features là nn.Sequential).
        Ví dụ: nếu num_layers=2, ta sẽ unfreeze 2 block cuối cùng.
        """
        children = list(self.features.children())
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True

# ====================================================
# ==== EfficientNetB0_BiGRU_MambaSSM Model =========
# ====================================================
class EfficientNetB0_unfreeze_BiGRU_MambaSSM(nn.Module):
    """ 
    Mô hình dự đoán tâm đồng tử sử dụng:
    - EfficientNetB0_unfreeze làm Backbone cho phần trích xuất đặc trưng (với residual connection),
    - BiGRU để học quan hệ thời gian,
    - MambaSSM để xử lý chuỗi thời gian có chọn lọc,
    - Fully Connected để dự đoán tọa độ (x, y).
    
    Đầu vào: (batch_size, seq_len, channels, height, width)
    Đầu ra: (batch_size, seq_len, 2)
    """
    def __init__(self, args, feature_dim=256, mamba_hidden_dim=256, seq_len=10, gru_hidden_size=128):
        super(EfficientNetB0_unfreeze_BiGRU_MambaSSM, self).__init__()
        self.args = args
        self.backbone = EfficientNetBackbone_unfreeze(feature_dim=feature_dim, pretrained=True)
        self.bigru = nn.GRU(input_size=feature_dim, hidden_size=gru_hidden_size, 
                            num_layers=1, bidirectional=True, batch_first=True)
        # Input của MambaSSM được xác định là gru_hidden_size*2 (đầu ra của GRU)
        self.mamba = MambaSSM(input_dim=gru_hidden_size*2, hidden_dim=mamba_hidden_dim, seq_len=seq_len)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(mamba_hidden_dim, 2)

    def forward(self, x):
        """
        x: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        # Xử lý từng frame qua Backbone
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # Xử lý chuỗi qua GRU
        x, _ = self.bigru(x)  # (batch_size, seq_len, gru_hidden_size*2)
        # Xử lý chuỗi qua MambaSSM
        x = self.mamba(x)     # (batch_size, seq_len, mamba_hidden_dim)
        x = self.dropout(x)
        # Dự đoán tọa độ (x, y) cho mỗi khung hình
        x = self.fc(x)        # (batch_size, seq_len, 2)
        return x

    def unfreeze_backbone(self, num_layers=1):
        """
        Gọi hàm này sau vài epoch để unfreeze num_layers cuối cùng của backbone nhằm fine-tune.
        """
        self.backbone.unfreeze_layers(num_layers=num_layers)
