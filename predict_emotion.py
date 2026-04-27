
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load trained model
# =========================
model_path = "./final_emotion_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# =========================
# Load labels
# =========================
labels_list = load_dataset("go_emotions")["train"].features["labels"].feature.names

# =========================
# Sample texts
# =========================
texts = [
    "RCB completely crushed CSK, and it was embarrassing to watch🤣🤣.",
    "I am so happy today!",
    "I feel scared walking alone",
    "This made me really angry",
    "Thanks for helping me so much",
    "Are u fuking craze, have you got mad",
    "I got fustrated, so i went home early yesterday",
    "The toddler let out a squeal of delight when he saw the colorful balloons floating toward him  .",
    "She watched the taillights of the car fade into the distance, knowing it would be years before they met again.",
    "A low, guttural growl echoed from the shadows of the cave, making every hair on his neck stand up.",
    "He wrinkled his nose and recoiled as the pungent, sour smell of the spoiled milk hit him.",
    "Her jaw dropped and her eyes widened when she walked into the room to find all her friends shouting 'Happy Birthday!'.",
    "His face turned a deep shade of crimson as he slammed his fist onto the desk, tired of being ignored."

]

# =========================
# Tokenize
# =========================
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64
).to(device)

# =========================
# Predict
# =========================
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)

    preds = torch.argmax(probs, dim=1)
    confidences = torch.max(probs, dim=1).values

# =========================
# Output
# =========================
print("\n🎯 Final Predictions:\n")

for i, text in enumerate(texts):
    emotion = labels_list[preds[i]]
    confidence = confidences[i].item()

    print(f"Text: {text}")
    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 50)