# AI-Driven Video Captioning 

This project demonstrates AI-driven video captioning and summarization. The model extracts key features from video frames using a CNN-based encoder and generates captions with an LSTM-based or Transformer-based decoder. The project also includes techniques for video summarization by selecting important frames.

## Project Overview
The goal of this project is to create a system that can:
- Automatically generate captions for videos.
- Summarize videos by selecting the most relevant frames and generating concise summaries.

## Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/AI-Driven-Video-Captioning.git
    cd AI-Driven-Video-Captioning
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your video dataset by placing it in the `data/video_dataset/` folder.

## Usage
Run the `main.py` script to train the model:
```bash
python main.py
