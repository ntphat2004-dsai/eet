import torch
import torch.nn as nn
import torch.functional as F

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

# Module trích xuất đặc trưng cho từng frame
class FrameFeatureExtractor(nn.Module):
    def __init__(self, channels, height, width):
        super(FrameFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        # Sau 2 lần MaxPool, chiều cao và chiều rộng giảm xuống còn 1/4
        self.feature_dim = 64 * (height // 4) * (width // 4)

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)  # flatten thành (B, feature_dim)
        return out

# Mô hình BiGRU kết hợp Self-Attention với tham số đầu vào từ args
class BiGRU_AttentionModel(nn.Module):
    def __init__(self, args):
        super(BiGRU_AttentionModel, self).__init__()
        self.args = args
        # Lấy các tham số từ args
        self.n_time_bins = args.n_time_bins    # số bước thời gian (seq_len)
        self.height = args.height              # chiều cao của frame
        self.width = args.width                # chiều rộng của frame
        self.channels = args.channels if hasattr(args, 'channels') else 1
        self.num_gru_units = args.num_gru_units if hasattr(args, 'num_gru_units') else 64

        # Sử dụng module FrameFeatureExtractor để trích xuất đặc trưng của từng frame
        self.feature_extractor = FrameFeatureExtractor(self.channels, self.height, self.width)
        self.gru_input_size = self.feature_extractor.feature_dim

        # BiGRU xử lý chuỗi các đặc trưng
        self.bi_gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.num_gru_units, 
                             batch_first=True, bidirectional=True)
        
        # Lớp Self-Attention tính điểm cho mỗi bước thời gian
        self.attention_layer = nn.Sequential(
            nn.Linear(2 * self.num_gru_units, 1),
            nn.Tanh()
        )
        # Lớp Fully-connected dự đoán tọa độ (x, y)
        self.fc = nn.Linear(2 * self.num_gru_units, 2)

    def forward(self, x):
        # x có shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        # Ghép batch và time để xử lý từng frame riêng biệt: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        # Trích xuất đặc trưng từ từng frame
        features = self.feature_extractor(x)  # (B*T, feature_dim)
        # Xếp lại thành chuỗi: (B, T, feature_dim)
        features = features.view(B, T, -1)
        
        # Xử lý chuỗi đặc trưng qua BiGRU
        gru_out, _ = self.bi_gru(features)  # (B, T, 2*num_gru_units)
        
        # Tính điểm attention cho từng bước thời gian
        attn_scores = self.attention_layer(gru_out)  # (B, T, 1)
        attn_scores = attn_scores.squeeze(-1)         # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)    # (B, T)
        
        # Tính vector context theo trọng số attention
        attn_weights = attn_weights.unsqueeze(-1)         # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # (B, 2*num_gru_units)
        
        # Dự đoán tọa độ (x, y)
        output = self.fc(context)  # (B, 2)
        return output

