# teacher_train.py
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 2. Load GoEmotions dataset
# =========================
dataset = load_dataset("go_emotions")
dataset = dataset["train"].train_test_split(test_size=0.1)

train_ds = dataset["train"]
eval_ds = dataset["test"]

labels_list = train_ds.features["labels"].feature.names
num_labels = len(labels_list)
print("Number of emotions:", num_labels)
print(labels_list)

# =========================
# 3. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

# =========================
# 4. Convert multi-label to single-label
# =========================
def to_single_label(example):
    example["label"] = example["labels"][0] if len(example["labels"]) > 0 else 27
    return example

train_ds = train_ds.map(to_single_label)
eval_ds = eval_ds.map(to_single_label)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =========================
# 5. Load model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
).to(device)

# =========================
# 6. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

dataset_size = len(train_ds)
batch_size = 16
steps_per_epoch = dataset_size // batch_size

eval_steps = max(1, steps_per_epoch // 5)  # evaluate 5 times per epoch

# =========================
# 7. TRAINING ARGUMENTS ✅ FIXED
# =========================
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,   # ✅ reduced
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # helps simulate larger batch
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),  # ✅ FIXED
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=1,
    report_to="none"
)



#=======================
# 8. Data collator
# =========================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 9. Trainer
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
# 10. Train & save
# =========================
print("🚀 Training Teacher model...")
trainer.train()
trainer.save_model("./teacher_model")
tokenizer.save_pretrained("./teacher_model")
print("✅ Teacher model saved at ./teacher_model")