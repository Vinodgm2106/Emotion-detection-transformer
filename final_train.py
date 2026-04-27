


import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, Features, Value, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import torch

# =========================
# 1. Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 2. Load pseudo-labeled dataset (CSV)
# =========================
csv_path = "pseudo_labeled_1M.csv"
df_custom = pd.read_csv(csv_path)
df_custom = df_custom.dropna(subset=["text", "emotion_label"])

# Rename column for consistency
df_custom = df_custom.rename(columns={"emotion_label": "label"})

# =========================
# 3. Load GoEmotions dataset (single-label)
# =========================
go_ds = load_dataset("go_emotions")["train"]

def to_single_label(example):
    example["label"] = example["labels"][0] if len(example["labels"]) > 0 else 27
    return example

go_ds = go_ds.map(to_single_label)
go_ds = go_ds.remove_columns(["labels", "id"])

# =========================
# 4. Define unified features
# =========================
features = Features({
    "text": Value("large_string"),
    "label": Value("int64")
})

# Create custom dataset
custom_ds = Dataset.from_pandas(
    df_custom[["text", "label"]],
    features=features,
    split="train"
)

# Cast GoEmotions dataset
go_ds = go_ds.cast(features)

# =========================
# 5. Concatenate datasets
# =========================
combined_ds = concatenate_datasets([go_ds, custom_ds])
print("✅ Combined dataset size:", len(combined_ds))

# =========================
# 6. Tokenization
# =========================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=64)

combined_ds = combined_ds.train_test_split(test_size=0.1)
train_ds = combined_ds["train"].map(tokenize, batched=True)
eval_ds = combined_ds["test"].map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =========================
# 7. Model
# =========================
num_labels = 28
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
).to(device)

# =========================
# 8. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# =========================
# 9. Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./final_emotion_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,   # doubled for A4000
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=True,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",       # weighted F1 better for multi-class
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=1,
    report_to="none"
)
# =========================
# 10. Data collator
# =========================
data_collator = DataCollatorWithPadding(tokenizer)

# =========================
# 11. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# 12. Train & Save
# =========================
print("🚀 Training Final 28-class Emotion model...")
trainer.train()

trainer.save_model("./final_emotion_model")
tokenizer.save_pretrained("./final_emotion_model")
print("✅ Final model saved at ./final_emotion_model")