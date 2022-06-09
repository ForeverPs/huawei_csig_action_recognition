import torch
import torch.nn as nn
from model.ml_decoder import MLDecoder
from model.temporal_module import build_base_model


# class ActNet(nn.Module):
#     def __init__(self, ActionLength=800, p=0.2):
#         super(ActNet, self).__init__()
#         self.map = nn.Sequential(
#             self.cls_block(24 * 6, 256, p),
#             self.cls_block(256, 512, p),
#             nn.Linear(512, 1024)
#         )
        
#         self.pool = nn.AdaptiveAvgPool1d(ActionLength)
#         # self.lstm = build_base_model(base_type='bilstm', num_feature=512, num_head=8)
#         self.lstm = build_base_model(base_type='attention', num_feature=1024, num_head=8)
#         self.decoder = MLDecoder(num_classes=10, initial_num_features=1024)
    
#     def cls_block(self, channel_in, channel_out, p):
#         block = nn.Sequential(
#             nn.Linear(channel_in, channel_out),
#             nn.GELU(),
#             nn.Dropout(p),
#             nn.LayerNorm(channel_out),
#         )
#         return block

#     def forward(self, x):
#         # x : uncertain frames, 24, 6
#         x = x.reshape(x.shape[0], -1) # frames, 24 * 6
#         x = self.map(x)  # frames, 24 * 6 -> frames, 1024
#         x = self.pool(x.permute(1, 0)).permute(1, 0).unsqueeze(0)  # 1, 800, 1024
#         x = x + self.lstm(x) # 1, 800, 512
#         x = self.decoder(x)
#         return x


class ActNet(nn.Module):
    def __init__(self, ActionLength=800):
        super(ActNet, self).__init__()
        self.map = nn.Linear(24 * 6, 1280)
        
        self.pool = nn.AdaptiveAvgPool1d(ActionLength)
        # self.lstm = build_base_model(base_type='bilstm', num_feature=512, num_head=8)
        self.lstm = build_base_model(base_type='attention', num_feature=1280, num_head=8)
        self.decoder = MLDecoder(num_classes=10, initial_num_features=1280)

    def forward(self, x):
        # x : uncertain frames, 24, 6
        x = x.reshape(x.shape[0], -1)  # frames, 24 * 6
        x = self.map(x)  # frames, 24 * 6 -> frames, 1024
        x = self.pool(x.permute(1, 0)).permute(1, 0).unsqueeze(0)  # 1, 800, 1024
        x = x + self.lstm(x)  # 1, 800, 512
        x = self.decoder(x)
        return x
