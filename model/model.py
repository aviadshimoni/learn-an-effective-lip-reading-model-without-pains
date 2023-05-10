from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler


class VideoModel(nn.Module):

    def __init__(self, num_classes, dropout=0.5):
        super(VideoModel, self).__init__()

        self.num_classes = num_classes

        self.video_cnn = VideoCNN()
        in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)

        self.v_cls = nn.Linear(1024 * 2, self.num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        h, _ = self.gru(f_v)

        y_v = self.v_cls(self.dropout(h)).mean(1)

        return y_v
