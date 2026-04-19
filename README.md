# 🎭 Emotion Recognition System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance emotion recognition system that detects **6 emotions** from text using fine-tuned RoBERTa, achieving **94.45% accuracy** - surpassing the paper's 88% benchmark.

## 🎯 Emotions Detected

| Emotion | Emoji | Example |
|---------|-------|---------|
| Sadness | 😢 | "I feel so lonely today" |
| Joy | 😊 | "I'm so excited about this!" |
| Love | ❤️ | "I adore spending time with you" |
| Anger | 😠 | "This is absolutely infuriating!" |
| Fear | 😨 | "I'm terrified of what might happen" |
| Surprise | 😲 | "Wow! I didn't expect that!" |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **94.45%** |
| **F1 Score** | **94.47%** |
| Precision | 94.45% |
| Recall | 94.45% |

**Comparison with Paper (Pajon-Sanmartin et al., 2025):**
- Paper: 88.00% accuracy
- Our model: **94.45% accuracy**
- **Improvement: +6.45%** 🎉

## 🚀 Quick Start

1. Clone the repository
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition

2. Install dependencies
pip install -r requirements.txt

3. Download the model
# Option A: Download from Hugging Face (coming soon)
# Option B: Train your own model
python training/train_model.py

4. Run the web app
bash
streamlit run app/streamlit_app.py
