import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
import os
import numpy as np

# 20D 감정 라벨 (예시, 실제 라벨 리스트와 일치시켜야 함)
labels_20d = [
    "admiration", "amusement", "anger", "compassion", "contempt",
    "contentment", "disappointment", "disgust", "fear", "guilt",
    "hate", "interest", "joy", "love", "pleasure",
    "pride", "regret", "relief", "sadness", "shame"
]



model_path = "https://github.com/Phoo1911/color-recommendation-api/tree/d7200f0ddae15572c06335196b836f6875c3fd5a/goemotions-lora-llama3-20d"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    num_labels=20,
    device_map="cuda:2",
    torch_dtype=torch.float32,
    problem_type="multi_label_classification"
)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
device = model.device if hasattr(model, "device") else torch.device("cuda:2")

def get_emotion_vector(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    top_ids = probs.argsort()[-2:][::-1]
    top_labels = [labels_20d[i] for i in top_ids]
    top_confidence = float(probs[top_ids[0]])

    class_probabilities = {
        labels_20d[i]: round(float(probs[i]), 4) for i in range(len(labels_20d))
    }

    return {
        "text": text,
        "emotion_top2": top_labels,
        "confidence": round(top_confidence, 4),
        "class_probabilities": class_probabilities,
        "emotion_vector": [round(float(p), 4) for p in probs.tolist()]
    }

# emotion_infer.py에 추가할 함수

def get_top_emotions_with_confidence(text: str):
    """
    기존 get_emotion_vector 함수를 활용해서 필요한 형태로 반환
    """
    result = get_emotion_vector(text)
    
    emotion_20d = result["emotion_vector"]  # 20차원 벡터
    top_emotions = result["emotion_top2"]   # 상위 2개 감정
    confidence = result["confidence"]       # 신뢰도
    class_probs = result["class_probabilities"]  # 클래스별 확률
    
    return emotion_20d, top_emotions, confidence, class_probs
    
    return emotion_20d, top_emotions, confidence, class_probs
# ✅ 테스트 예시
if __name__ == "__main__":
    sample_text = "I feel sad but also hopeful today."
    result = get_emotion_vector(sample_text)
    from pprint import pprint
    pprint(result)
