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

class CNN_BiGRU_SelfAttention(nn.Module):
    """
    Mô hình dự đoán tâm đồng tử kết hợp CNN + BiGRU + SelfAttention với Dropout.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # Thêm dropout sau pooling
        self.dropout = nn.Dropout(p=0.5)
        # Sử dụng BiGRU với hidden_size = 128 -> output có kích thước 256
        self.bigru = nn.GRU(input_size=36192, hidden_size=128, num_layers=1, 
                            bidirectional=True, batch_first=True)
        self.self_attention = SelfAttention(input_dim=256)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        # x có shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        # Đảo vị trí chiều height và width nếu cần
        x = x.permute(0, 1, 3, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)  # Áp dụng dropout sau pooling

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.bigru(x)  # output có shape (batch_size, seq_len, 256)
        x = self.self_attention(x)
        x = self.fc(x)  # output cuối có shape (batch_size, seq_len, 2)
        return x

