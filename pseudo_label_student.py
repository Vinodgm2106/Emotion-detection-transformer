import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# =========================
# 1. Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 2. Load teacher model
# =========================
model_path = "./teacher_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# =========================
# 3. Load your CSV dataset (3M)
# =========================
csv_path = r"/home/cnsdev/streamlit_python_files/vi/dataset/dataset_cleaned_1.csv"
df = pd.read_csv(csv_path)
df = df.sample(1_000_000, random_state=42)  # sample 1M rows
df = df.dropna(subset=["text"])
print(f"Dataset size: {len(df)}")

# =========================
# 4. Batch prediction for pseudo-labeling
# =========================
batch_size = 64
all_preds = []

for i in range(0, len(df), batch_size):
    texts = df["text"].iloc[i:i+batch_size].tolist()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

    with torch.no_grad():   
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    all_preds.extend(preds)

df["emotion_label"] = all_preds
df.to_csv("pseudo_labeled_1M.csv", index=False)
print("✅ Pseudo-labeled dataset saved at pseudo_labeled_1M.csv")