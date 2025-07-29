import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# 💡 1. MLP 모델 구조 및 로드
# -----------------------------
class TunedMLP(nn.Module):
    def __init__(self, input_dim=20, output_dim=3):
        super(TunedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def load_model(model_path, input_dim=20):
    model = TunedMLP(input_dim=input_dim, output_dim=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_rgb_mlp(emotion_vec: np.ndarray, model: nn.Module):
    with torch.no_grad():
        input_tensor = torch.tensor(emotion_vec, dtype=torch.float32).unsqueeze(0)  # shape: [1, 20]
        output = model(input_tensor).squeeze().numpy()  # shape: [3]
        output = np.clip(output, 0, 1)  # RGB는 0~1 범위로 제한
    return output

# -----------------------------
# 💡 2. 테스트 실행
# -----------------------------
if __name__ == "__main__":
    # 예시 입력 (20차원 감정 벡터)
    sample_input = np.random.rand(20)

    # MLP 추론
    model = load_model("/home/tmp1/kyu/emotion_color_api/models/weights/mlp_model_tuned.pt", input_dim=20)
    rgb_mlp = predict_rgb_mlp(sample_input, model)
    print("🎨 MLP RGB:", rgb_mlp)
