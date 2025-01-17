import torch
import torch.nn as nn
import torchvision.models as models

class VideoEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VideoEncoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove final layer
        self.fc = nn.Linear(resnet.fc.in_features, 256)  # Project features to 256 dims

    def forward(self, x):
        x = self.features(x)  # Extract features from image
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Project to desired dimension
        return x
