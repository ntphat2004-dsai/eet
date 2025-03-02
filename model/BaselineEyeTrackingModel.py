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
        # Kích thước sau 2 lần MaxPool (giả sử height, width chia hết cho 4)
        self.feature_dim = 64 * (height // 4) * (width // 4)

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        return out

# Mô hình BiGRU kết hợp Self-Attention
class BiGRU_AttentionModel(nn.Module):
    def __init__(self, n_time_bins, height, width, channels=1, num_gru_units=64):
        super(BiGRU_AttentionModel, self).__init__()
        self.n_time_bins = n_time_bins
        self.feature_extractor = FrameFeatureExtractor(channels, height, width)
        self.gru_input_size = self.feature_extractor.feature_dim
        self.bi_gru = nn.GRU(input_size=self.gru_input_size, hidden_size=num_gru_units, 
                             batch_first=True, bidirectional=True)
        self.attention_layer = nn.Sequential(
            nn.Linear(2 * num_gru_units, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(2 * num_gru_units, 2)  # Dự đoán tọa độ (x, y)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        # Ghép batch và time để xử lý từng frame
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, feature_dim)
        features = features.view(B, T, -1)      # (B, T, feature_dim)
        
        # Xử lý chuỗi bằng BiGRU
        gru_out, _ = self.bi_gru(features)  # (B, T, 2*num_gru_units)
        
        # Tính điểm attention cho mỗi bước thời gian
        attn_scores = self.attention_layer(gru_out)  # (B, T, 1)
        attn_scores = attn_scores.squeeze(-1)         # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)    # (B, T)
        
        # Tính vector context theo trọng số attention
        attn_weights = attn_weights.unsqueeze(-1)      # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # (B, 2*num_gru_units)
        
        # Dự đoán tọa độ
        output = self.fc(context)  # (B, 2)
        return output

# # Ví dụ khởi tạo mô hình và chạy qua 1 batch dummy data
# if __name__ == "__main__":
#     n_time_bins = 30   # Số frame theo thời gian
#     height = 64        # Chiều cao mỗi frame
#     width = 64         # Chiều rộng mỗi frame
#     channels = 1       # Số kênh (ví dụ ảnh xám)

#     model = BiGRU_AttentionModel(n_time_bins, height, width, channels, num_gru_units=64)
#     print(model)

#     # Dummy input: (batch_size, n_time_bins, channels, height, width)
#     dummy_input = torch.randn(8, n_time_bins, channels, height, width)
#     output = model(dummy_input)
#     print("Output shape:", output.shape)  # Dự kiến (8, 2)
