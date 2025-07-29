import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

personal_color_palettes = {
    "Spring_Light": [
        [0.980, 0.839, 0.647],
        [1.000, 0.854, 0.725],
        [1.000, 0.980, 0.804],
        [0.901, 0.901, 0.980],
        [0.941, 0.901, 0.549]
    ],
    "Summer_Cool": [
        [0.8, 0.8, 1.0],
        [0.9, 0.85, 1.0],
        [0.75, 0.9, 1.0],
        [0.85, 0.95, 0.95],
        [0.95, 0.9, 1.0]
    ],
    # 추가 palette 정의 가능
}

def get_best_palette_color(emotion_rgb: np.ndarray, personal_color_type: str):
    palette_rgb = np.array(personal_color_palettes.get(personal_color_type, []))
    if palette_rgb.shape[0] == 0:
        return emotion_rgb, "#%02x%02x%02x" % tuple((emotion_rgb * 255).astype(int)), []

    sims = cosine_similarity([emotion_rgb], palette_rgb)[0]
    best_idx = np.argmax(sims)
    best_rgb = palette_rgb[best_idx]
    best_rgb255 = (best_rgb * 255).astype(int)
    hex_color = '#%02x%02x%02x' % tuple(best_rgb255)

    palette_hex = ['#%02x%02x%02x' % tuple((c * 255).astype(int)) for c in palette_rgb]

    return best_rgb, hex_color, palette_hex
