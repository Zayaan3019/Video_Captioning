import torch
from torch.utils.data import DataLoader
from models.captioning_model import VideoCaptioningModel
from utils.data_loader import VideoDataset
from torch.optim import Adam

# Load the dataset
train_dataset = VideoDataset('data/video_dataset/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = VideoCaptioningModel(embed_size=256, hidden_size=512, vocab_size=5000)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for video_frames, captions in train_loader:
        optimizer.zero_grad()
        outputs = model(video_frames, captions)
        loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(2)), captions.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
