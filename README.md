# Emotion-detection-transformer
#  Large-Scale Emotion Detection using Transformers & Semi-Supervised Learning

##  Overview
This project builds a **high-performance emotion classification system** capable of detecting **28 human emotions** from text using advanced NLP techniques.

It leverages a **Transformer-based architecture (RoBERTa)** combined with a **Teacher-Student semi-supervised learning approach** to scale from a labeled dataset to **1M+ pseudo-labeled samples**, significantly improving performance.

---

##  Key Highlights
-  Built an end-to-end NLP pipeline using **HuggingFace Transformers**
-  Implemented **Teacher-Student pseudo-labeling framework**
-  Scaled dataset from **GoEmotions → 1M+ samples**
-  Handled **multi-class (28 emotions)** classification
-  Optimized using **Weighted F1-score**
-  Designed for **GPU training (FP16 enabled)**

---

##  Problem Statement
Emotion detection is a complex NLP task requiring understanding of context, tone, and subtle linguistic patterns.

This project aims to:
- Accurately classify emotions from raw text
- Improve performance using large-scale semi-supervised learning
- Build a scalable and production-ready pipeline

---

##  Architecture & Pipeline

### Step 1: Train Teacher Model
- Trained on **GoEmotions dataset**
- Converted multi-label → single-label classification
- Model: `RoBERTa-base`

### Step 2: Pseudo Labeling (Student Data Generation)
- Used trained teacher model to label **1M unlabeled text samples**
- Generated `pseudo_labeled_1M.csv`

### Step 3: Final Training
- Combined:
  - GoEmotions dataset
  - Pseudo-labeled dataset
- Trained final model on **large-scale dataset**

### Step 4: Inference
- Predicts:
  - Emotion label
  - Confidence score

---

## 📁 Project Structure
Emotion-Detection-Transformer/
│
├── README.md
├── requirements.txt
│
├── src/
│ ├── teacher_train.py # Train teacher model on GoEmotions
│ ├── pseudo_label_student.py # Generate pseudo labels for large dataset
│ ├── final_train.py # Train final model on combined dataset
│ └── predict_emotion.py # Run inference on new text
│
├── models/
│ ├── teacher_model/
│ └── final_emotion_model/
│
└── outputs/
└── logs/

---

##  Tech Stack
- **Programming:** Python  
- **Deep Learning:** PyTorch  
- **NLP:** HuggingFace Transformers, Datasets  
- **Data Processing:** Pandas, NumPy  
- **Evaluation:** Scikit-learn  
- **Hardware:** GPU (CUDA with FP16 support)

---

##  Dataset Details

### 🔹 GoEmotions Dataset
- Source: Google Research
- 28 emotion categories
- Used for initial supervised training

### 🔹 Custom Dataset
- Large unlabeled dataset (~3M rows)
- Sampled **1M rows**
- Pseudo-labeled using teacher model

---

##  Model Details

| Component        | Details                  |
|----------------|-------------------------|
| Base Model     | RoBERTa-base            |
| Task           | Multi-class classification |
| Classes        | 28 emotions             |
| Max Length     | 64 tokens               |
| Batch Size     | 16                      |
| Epochs         | 3                       |
| Learning Rate  | 2e-5                    |
| Optimization   | Weighted F1-score       |
| Precision      | FP16 (mixed precision)  |

---

##  Evaluation Metrics
- Accuracy
- Weighted F1-score (primary metric)

---

##  Sample Predictions
Text: I am so happy today!
Emotion: joy
Confidence: 0.94

Text: I feel scared walking alone
Emotion: fear
Confidence: 0.91
