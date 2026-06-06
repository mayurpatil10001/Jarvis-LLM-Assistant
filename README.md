# 🤖 JARVIS — Custom Transformer AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An Iron Man-inspired AI assistant built entirely from scratch — featuring a custom GPT-style Transformer trained locally, voice input/output, speech recognition, and a dark-themed Tkinter GUI.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Training](#-training-the-model) • [Usage](#-usage) • [Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

JARVIS is a fully custom-built conversational AI assistant that implements the entire deep learning pipeline from scratch — no pretrained LLM APIs, no wrappers. The Transformer model is built using raw PyTorch with multi-head self-attention, positional encoding, and GELU activations. A Tkinter GUI provides a sleek Iron Man-style dark interface with both text and voice interaction modes.

This project demonstrates end-to-end ML engineering: custom tokenization, model architecture, training loop, inference with top-k sampling, and real-time GUI integration.

---

## ✨ Features

### 🧠 Custom Transformer Model
- GPT-style decoder-only architecture built from scratch in PyTorch
- Multi-head self-attention with 8 heads
- 8 Transformer blocks with feed-forward layers (d_ff = 2048)
- GELU activation, LayerNorm, residual connections, dropout
- Sinusoidal positional encoding
- Top-k sampling with temperature control for text generation

### 🎙️ Voice Interaction
- Real-time speech recognition via Google Speech API (`speech_recognition`)
- Text-to-speech output with JARVIS-style male voice (`pyttsx3`)
- Configurable speech rate and volume
- Toggle-based push-to-listen mode

### 🖥️ Tkinter GUI
- Iron Man-inspired dark theme (black + green terminal aesthetic)
- Scrollable chat history with timestamps
- Text input with Enter key support
- Voice input toggle button
- Real-time status indicator
- Auto-save chat history to JSON on exit

### 📦 Custom Tokenizer
- Word-level tokenizer with frequency pruning (`min_freq=2`)
- Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
- Encode/decode pipeline for training and inference

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    JARVIS Architecture                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Text → TokenizerVocab → Token Embeddings          │
│                                      ↓                   │
│                          Positional Encoding             │
│                                      ↓                   │
│              ┌─────────────────────────────┐             │
│              │     Transformer Block × 8   │             │
│              │  ┌─────────────────────┐    │             │
│              │  │  Multi-Head         │    │             │
│              │  │  Self-Attention     │    │             │
│              │  │  (8 heads, d=512)   │    │             │
│              │  └──────────┬──────────┘    │             │
│              │        LayerNorm + Residual  │             │
│              │  ┌─────────────────────┐    │             │
│              │  │  Feed Forward       │    │             │
│              │  │  (GELU, d_ff=2048)  │    │             │
│              │  └──────────┬──────────┘    │             │
│              │        LayerNorm + Residual  │             │
│              └─────────────────────────────┘             │
│                                      ↓                   │
│                          Final LayerNorm                 │
│                                      ↓                   │
│                       Linear → Vocab Logits              │
│                                      ↓                   │
│                    Top-K Sampling + Temperature          │
│                                      ↓                   │
│                         Generated Response               │
└─────────────────────────────────────────────────────────┘
```

### Model Parameters

| Parameter | Value |
|---|---|
| Model Dimension (`d_model`) | 512 |
| Attention Heads (`n_heads`) | 8 |
| Transformer Layers (`n_layers`) | 8 |
| Feed-Forward Dim (`d_ff`) | 2048 |
| Max Sequence Length | 512 |
| Dropout | 0.1 |
| Activation | GELU |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, CPU works too)
- Microphone (for voice input)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/jarvis-llm-assistant.git
cd jarvis-llm-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch
torchvision
numpy
tqdm
speechrecognition
pyttsx3
pyaudio
nbimporter
```

> **Note for Linux users:** Install `portaudio` for PyAudio:
> ```bash
> sudo apt-get install portaudio19-dev python3-pyaudio
> ```

> **Note for macOS users:**
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

---

## 🧠 Training the Model

### Prepare Your Dataset

Create a text file or JSON with conversational data:

```
What is your name? I am JARVIS, your AI assistant.
How are you? I am functioning at optimal capacity.
Tell me about AI. Artificial intelligence is the simulation of human intelligence.
```

### Run Training

```python
# In jarvis_training.py, configure:
TRAINING_CONFIG = {
    "data_path": "training_data.txt",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8
}
```

```bash
python jarvis_training.py
```

Training outputs:
- `jarvis_model.pth` — trained model weights + vocab size
- `tokenizer.pkl` — fitted tokenizer

### Generation Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_length` | 50 | Maximum tokens to generate |
| `temperature` | 0.7 | Creativity (lower = more focused) |
| `top_k` | 50 | Top-k sampling pool size |

---

## 🚀 Usage

### Launch the GUI

```bash
python jarvis_gui.py
```

### Text Mode
1. Type your message in the input field
2. Press **Enter** or click **Send**
3. JARVIS responds in the chat and speaks the reply

### Voice Mode
1. Click **🎤 Listen**
2. Speak your query
3. Click **🛑 Stop** or wait for auto-detection
4. JARVIS processes and responds with voice output

### Chat History
Chat is automatically saved as `chat_history_YYYYMMDD_HHMMSS.json` when you close the app.

---

## 📁 Project Structure

```
jarvis-llm-assistant/
│
├── jarvis_training.py      # Transformer model + tokenizer + training loop
│   ├── TokenizerVocab      # Custom word-level tokenizer
│   ├── PositionalEncoding  # Sinusoidal positional encoding
│   ├── MultiHeadAttention  # Scaled dot-product multi-head attention
│   ├── FeedForward         # GELU feed-forward block
│   ├── TransformerBlock    # Full transformer block with residuals
│   └── JARVISModel         # Full model with generation method
│
├── jarvis_gui.py           # Tkinter GUI + voice I/O
│   ├── JARVISAssistant     # Speech recognition + TTS + model interface
│   └── JARVISGui           # Dark-themed chat interface
│
├── jarvis_model.pth        # Trained model weights (generated after training)
├── tokenizer.pkl           # Fitted tokenizer (generated after training)
├── training_data.txt       # Your training corpus
└── requirements.txt
```

---

## 🎯 Key Technical Highlights

For recruiters and reviewers — what makes this project stand out:

| Concept | Implementation |
|---|---|
| **Transformer from scratch** | No HuggingFace — pure PyTorch nn.Module |
| **Custom tokenizer** | Frequency-based vocab with special tokens |
| **Inference sampling** | Top-k + temperature — not greedy decoding |
| **Threading** | Non-blocking voice I/O with Python threading |
| **Weight initialization** | Proper `N(0, 0.02)` init following GPT paper |
| **GUI integration** | Real-time model inference inside Tkinter event loop |

---

## 📈 Roadmap

- [ ] BPE (Byte-Pair Encoding) tokenizer for better subword handling
- [ ] Training on larger conversational datasets (DailyDialog, OpenSubtitles)
- [ ] Wake word detection ("Hey JARVIS")
- [ ] Long-term memory with conversation context window
- [ ] REST API mode for headless deployment
- [ ] GPU training script with mixed precision (fp16)
- [ ] Export to ONNX for faster inference

---

## 🤝 Contributing

```bash
# Fork the repo and create a feature branch
git checkout -b feature/your-feature

# Make changes and commit
git commit -m "Add: your feature description"

# Push and open a Pull Request
git push origin feature/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- Inspired by *Attention Is All You Need* (Vaswani et al., 2017)
- Sampling strategy inspired by GPT-2 generation
- Iron Man's JARVIS for the concept and aesthetic

---

<div align="center">
  <b>"Sometimes you gotta run before you can walk." — Tony Stark</b><br><br>
  Built from scratch. No shortcuts. 🔴🟡
</div>
