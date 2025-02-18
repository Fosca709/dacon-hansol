import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-8) -> np.ndarray:
    # a,b : [B, N], output: [B,]
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b + eps)


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0
