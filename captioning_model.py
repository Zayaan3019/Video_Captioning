import torch
import torch.nn as nn
from .video_encoder import VideoEncoder
from .sequence_decoder import CaptionDecoder

class VideoCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(VideoCaptioningModel, self).__init__()
        self.encoder = VideoEncoder()
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)

    def forward(self, video_frames, captions):
        features = self.encoder(video_frames)
        outputs = self.decoder(features, captions)
        return outputs
