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


# =========================
# Formal/Casual 라벨 점수화
# =========================
# NOTE: 초기 버전은 라벨 기반 휴리스틱 점수(1/0/-1)만 사용합니다.
#       추후 가중치/컨텍스트 기반 보정 로직을 추가할 예정입니다.

# 포멀(+1) 후보 라벨
FORMAL_IDS = {
    0,   # shirt, blouse
    2,   # sweater
    3,   # cardigan
    4,   # jacket
    5,   # vest
    9,   # coat
    10,  # dress
    16,  # tie
    18,  # watch
    19,  # belt
    25,  # scarf
    29,  # lapel
    32,  # pocket
    34,  # buckle
    35,  # zipper
    28,  # collar
    21,  # tights, stockings
}

# 캐주얼(-1) 후보 라벨
CASUAL_IDS = {
    1,   # top, t-shirt, sweatshirt
    7,   # shorts
    14,  # hat
    15,  # headband, head covering, hair accessory
    17,  # glove
    20,  # leg warmer
    27,  # hood
    11,  # jumpsuit
}

# 중립(0) 라벨
NEUTRAL_IDS = {
    30,  # epaulette
    36,  # applique
    37,  # bead
    38,  # bow
    39,  # flower
    41,  # ribbon
    42,  # rivet
    43,  # ruffle
    44,  # sequin
    13,  # glasses
    24,  # bag, wallet
    8,   # skirt
}

# 클래스ID별 포멀 점수 매핑 (-1, 0, 1)
FORMALITY_SCORE = {
    **{cid: 1 for cid in FORMAL_IDS},
    **{cid: -1 for cid in CASUAL_IDS},
    **{cid: 0 for cid in NEUTRAL_IDS},
}


def get_formality_score(class_id: int) -> int:
    """클래스 ID의 포멀 점수 반환 (-1: 캐주얼, 0: 중립, 1: 포멀)

    정의되지 않은 라벨은 기본 0으로 처리합니다.
    """
    return int(FORMALITY_SCORE.get(class_id, 0))


def get_formality_label(score: int) -> str:
    """포멀 점수 레이블 텍스트 반환"""
    if score > 0:
        return "Formal"
    if score < 0:
        return "Casual"
    return "Neutral"