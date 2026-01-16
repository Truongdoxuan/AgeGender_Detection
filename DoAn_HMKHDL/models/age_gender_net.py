import torch
import torch.nn as nn
import timm


class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            features_only=False
        )

        feat_dim = self.backbone.num_features  # 1024

        # Shared feature
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Heads
        self.gender_head = nn.Linear(256, 1)

        self.age_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone.forward_features(x) 
        x = x.mean(dim=[2, 3])                  
        x = self.fc(x)                         

        gender_logits = self.gender_head(x)
        age_pred = self.age_head(x)

        return gender_logits, age_pred
