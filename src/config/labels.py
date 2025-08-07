"""
의류 라벨 정의
Fashionpedia 데이터셋 기반 의류 카테고리 정의
"""

# Fashionpedia 의류 라벨 매핑
FASHIONPEDIA_LABELS = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt", 
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
    26: "umbrella",
    27: "hood",
    28: "collar",
    29: "lapel",
    30: "epaulette",
    31: "sleeve",
    32: "pocket",
    33: "neckline",
    34: "buckle",
    35: "zipper",
    36: "applique",
    37: "bead",
    38: "bow",
    39: "flower",
    40: "fringe",
    41: "ribbon",
    42: "rivet",
    43: "ruffle",
    44: "sequin",
    45: "tassel"
}

# 주요 의류 카테고리 (분석 우선순위)
MAJOR_CLOTHING_CATEGORIES = {
    "상의": [0, 1, 2, 3, 4, 5, 9, 10, 11],  # shirt, top, sweater, cardigan, jacket, vest, coat, dress, jumpsuit
    "하의": [6, 7, 8],                      # pants, shorts, skirt
    "신발": [23],                           # shoe
    "가방": [24],                           # bag, wallet
    "액세서리": [13, 14, 16, 18, 19, 25]    # glasses, hat, tie, watch, belt, scarf
}

# 색상 분석 제외 카테고리 (구조적 요소)
EXCLUDE_FROM_COLOR_ANALYSIS = [
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45
]  # hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper, applique, bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel

def get_label_name(class_id: int) -> str:
    """클래스 ID로 라벨명 반환"""
    return FASHIONPEDIA_LABELS.get(class_id, f"Unknown({class_id})")

def is_major_clothing(class_id: int) -> bool:
    """주요 의류 카테고리인지 확인"""
    for category_items in MAJOR_CLOTHING_CATEGORIES.values():
        if class_id in category_items:
            return True
    return False

def should_analyze_color(class_id: int) -> bool:
    """색상 분석 대상인지 확인"""
    return class_id not in EXCLUDE_FROM_COLOR_ANALYSIS