from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from models.emotion import get_emotion_vector
from models.color_mlp import predict_rgb_mlp, load_model
import numpy as np
import tempfile
import os
import requests
from pyngrok import ngrok
import uvicorn

app = FastAPI(
    title="Color Recommendation API",
    version="0.1.0",
    description="퍼스널 컬러 분석 및 감정 기반 색상 추천 API입니다.",
    docs_url="/color-docs",         # Swagger UI URL
    redoc_url="/redoc",       # ReDoc 문서 URL
)
    

# MLP 모델 로드 (전역 변수로 한 번만 로드)
try:
    mlp_model = load_model("/home/tmp1/kyu/emotion_color_api/models/weights/mlp_model_tuned.pt", input_dim=20)
    print("MLP 모델 로드 완료")
except Exception as e:
    print(f"MLP 모델 로드 실패: {e}")
    mlp_model = None

def predict_rgb_from_emotion(emotion_20d):
    """
    20차원 감정 벡터를 RGB로 변환
    """
    if mlp_model is None:
        # 모델이 없으면 기본값 반환
        return np.array([0.5, 0.5, 0.5])
    
    emotion_array = np.array(emotion_20d, dtype=np.float32)
    return predict_rgb_mlp(emotion_array, mlp_model)

class ColorRequest(BaseModel):
    text: str
    personal_color_type: Optional[str] = None
class EmotionOnlyRequest(BaseModel):
    text: str

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

# 기존 API는 그대로 유지
@app.post("/emotion_only_color")
def emotion_only_color(req: EmotionOnlyRequest):
    # Step 1. 감정 분석
    result = get_emotion_vector(req.text)
    top_emotions = result["emotion_top2"]
    emotion_20d = result["emotion_vector"]
    confidence = result["confidence"]
    class_probs = result["class_probabilities"]

    # Step 2. RGB 예측
    rgb_pred = predict_rgb_mlp(np.array(emotion_20d, dtype=np.float32), mlp_model)
    rgb_255 = (rgb_pred * 255).astype(int).tolist()
    hex_color = '#%02x%02x%02x' % tuple(rgb_255)

    # 결과 반환
    return {
        "mode": "emotion_only",
        "emotion_top2": top_emotions,
        "hex_color": hex_color,
        "rgb_255": rgb_255,
        "confidence": confidence,
        "class_probabilities": class_probs
    }

# 새로운 API: 이미지 업로드 + 텍스트로 퍼스널컬러 분석
def get_personal_color_from_api(image_path: str, api_url: str = "http://203.241.228.97:1111/predict_skin_tone"):
    """
    퍼스널컬러 예측 API 호출
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API 호출 실패: {response.status_code}")
            return None
    except Exception as e:
        print(f"퍼스널컬러 API 호출 오류: {e}")
        return None

def get_best_palette_color_from_api(emotion_rgb: np.ndarray, image_path: str):
    """
    API에서 받아온 팔레트로 최적 색상 찾기
    기존 get_best_palette_color와 같은 시그니처 유지
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # API 호출
    personal_color_data = get_personal_color_from_api(image_path)
    
    if not personal_color_data:
        # API 실패시 원본 감정 색상 반환
        emotion_hex = "#%02x%02x%02x" % tuple((emotion_rgb * 255).astype(int))
        return emotion_rgb, emotion_hex, []
    
    # hex_palette를 RGB로 변환
    palette_hex_list = personal_color_data.get("hex_palette", [])
    
    if not palette_hex_list:
        # 팔레트 없으면 원본 감정 색상 반환
        emotion_hex = "#%02x%02x%02x" % tuple((emotion_rgb * 255).astype(int))
        return emotion_rgb, emotion_hex, []
    
    # hex를 RGB 배열로 변환
    def hex_to_rgb_normalized(hex_color: str) -> np.ndarray:
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    
    palette_rgb = np.array([hex_to_rgb_normalized(hex_color) for hex_color in palette_hex_list])
    
    # cosine similarity로 가장 유사한 색상 찾기
    sims = cosine_similarity([emotion_rgb], palette_rgb)[0]
    best_idx = np.argmax(sims)
    best_rgb = palette_rgb[best_idx]
    
    # 최적 색상을 hex로 변환
    best_rgb255 = (best_rgb * 255).astype(int)
    hex_color = '#%02x%02x%02x' % tuple(best_rgb255)
    
    return best_rgb, hex_color, palette_hex_list, personal_color_data.get("predicted_label", "Unknown")

@app.post("/recommend_color_with_image")
async def recommend_color_with_image(
    text: str, 
    image: UploadFile = File(...)
):
    """
    텍스트 + 이미지로 퍼스널컬러 기반 색상 추천
    """
    # Step 1. 임시 파일로 이미지 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await image.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Step 2. 감정 분석
        emotion_20d, top_emotions, confidence, class_probs = get_top_emotions_with_confidence(text)
        
        # Step 3. 감정 -> RGB 변환
        emotion_rgb = predict_rgb_from_emotion(emotion_20d)
        
        # Step 4. 퍼스널컬러 API로 팔레트 받아서 매칭
        best_rgb, best_hex, palette_hex, personal_color_type = get_best_palette_color_from_api(emotion_rgb, temp_path)
        
        # Step 5. 원본 감정 색상도 함께 제공
        emotion_hex = '#%02x%02x%02x' % tuple((emotion_rgb * 255).astype(int))
        
        return {
            "mode": "emotion_plus_personal_color_api",
            "emotion_top2": top_emotions,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "original_emotion_hex": emotion_hex,
            "personal_color_type": personal_color_type,
            "palette_hex": palette_hex,
            "recommended_hex": best_hex,
        }
    
    finally:
        # 임시 파일 삭제
        os.unlink(temp_path)

# 헬스체크 엔드포인트
@app.get("/")
def health_check():
    return {
        "status": "API is running",
        "endpoints": [ "/recommend_color_with_image", "/emotion_only_color"]
    }

# ✅ main 함수로 구성 (ngrok 먼저 실행)
if __name__ == "__main__":
    # 1. ngrok으로 public URL 연결 (http:// 또는 https://로 시작함)
    public_url = ngrok.connect(1911)
    print("🚀 Public URL:", public_url)

    # 2. FastAPI 실행 (host=0.0.0.0 필수)
    uvicorn.run(app, host="0.0.0.0", port=1911, log_level="info")
