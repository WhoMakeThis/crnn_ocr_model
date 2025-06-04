import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, H, W)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (B, 64, H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (B, 128, H/4, W/4)
        )

        # RNN expects input of shape (batch, sequence_len=W, features=128*H/4)
        self.rnn = nn.LSTM(
            input_size=128 * (imgH // 4),  # 128 * 12 = 1536
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(nh * 2, nclass)  # bidirectional => nh * 2

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x)  # -> (B, C, H', W')
        b, c, h, w = conv.size()

        # ✅ 수정된 높이 체크
        assert h == 12, f"Expected height=12 after conv, got {h}"

        # prepare for RNN: (B, W, C*H)
        conv = conv.permute(0, 3, 1, 2)       # (B, W, C, H)
        conv = conv.contiguous().view(b, w, c * h)  # (B, W, C*H)

        rnn_out, _ = self.rnn(conv)  # -> (B, W, nh*2)
        output = self.fc(rnn_out)   # -> (B, W, nclass)
        output = output.permute(1, 0, 2)  # -> (W, B, nclass) for CTC Loss

        return output
