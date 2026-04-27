# Emotion-detection-transformer
# 🔥 Large-Scale Emotion Detection using Transformers & Semi-Supervised Learning

## 🚀 Overview
This project builds a **high-performance emotion classification system** capable of detecting **28 human emotions** from text using advanced NLP techniques.

It leverages a **Transformer-based architecture (RoBERTa)** combined with a **Teacher-Student semi-supervised learning approach** to scale from a labeled dataset to **1M+ pseudo-labeled samples**, significantly improving performance.

---

## 🧠 Key Highlights
- ✅ Built an end-to-end NLP pipeline using **HuggingFace Transformers**
- ✅ Implemented **Teacher-Student pseudo-labeling framework**
- ✅ Scaled dataset from **GoEmotions → 1M+ samples**
- ✅ Handled **multi-class (28 emotions)** classification
- ✅ Optimized using **Weighted F1-score**
- ✅ Designed for **GPU training (FP16 enabled)**

---

## 📊 Problem Statement
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
