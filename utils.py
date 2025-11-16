"""애플리케이션 유틸리티 함수"""
import logging
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
import os
import pandas as pd
import numpy as np
from geocoding import get_admin_district_from_coords

# cluster.py에서 행정동 매칭 함수 가져오기
# 순환 참조를 피하기 위해 함수 내에서 import 할 수도 있지만, 구조상 utils는 cluster보다 하위 레벨이므로 직접 import
try:
    from cluster import assign_reports_to_admin_districts
except ImportError:
    # 만약 다른 환경에서 utils만 단독으로 쓰일 경우를 대비
    assign_reports_to_admin_districts = None

logger = logging.getLogger(__name__)


def calculate_priority_score(damage_severity, population_weight, traffic_ratios, 
                             weights=(0.5, 0.25, 0.25)):
    """
    공공시설 파손 우선순위 점수 계산
    
    Parameters:
    - damage_severity (float): 파손 심각도 점수 (1~5)
    - population_weight (float): 지역/시간별 인구 가중치 (raw 값, 0~)
    - traffic_ratios (list): 100m 내 교차로 차량수 비율 리스트 (raw 값, 0~)
    - weights (tuple): (w1, w2, w3) 가중치 (파손 심각도, 인구 가중치, 교통량)
    
    Returns:
    - float: 최종 우선순위 점수 (1~5 범위)
    """
    
    # A: 파손 상태 점수 (그대로 사용)
    A = damage_severity
    
    # B: 인구 가중치 스케일링 1~5 
    B_scaled = 1 + 4 * population_weight
    
    # C: 교차로 차량수 비율 스케일링 1~5
    if traffic_ratios and len(traffic_ratios) > 0:
        # 100m 내 교차로들의 평균 사용
        C_scaled = 1 + 4 * np.mean(traffic_ratios)
    
    # 개별 스케일링된 긴급도 로깅
    logger.info(f"Scaled urgency components: Damage(A)={A:.2f}, Population(B)={B_scaled:.2f}, Traffic(C)={C_scaled:.2f}")

    # 최종 가중 평균 계산
    w1, w2, w3 = weights
    final_score = (w1 * A + w2 * B_scaled + w3 * C_scaled)
    
    # 최종 점수가 1~5 범위를 벗어나지 않도록 보정
    return max(1.0, min(final_score, 5.0))



def extract_location(image_bytes: bytes) -> dict:
    """이미지 EXIF 데이터에서 위치 정보 추출"""
    try:
        image = Image.open(BytesIO(image_bytes))
        
        # EXIF 데이터 안전 처리
        try:
            exif = image._getexif()
            if exif:
                # GPS 정보 추출
                gps_info = exif.get(34853)
                if gps_info:
                    try:
                        lat_raw = gps_info.get(2)
                        lon_raw = gps_info.get(4)
                        
                        # 튜플(도, 분, 초)을 도로 변환
                        if isinstance(lat_raw, tuple) and len(lat_raw) >= 3:
                            lat = lat_raw[0] + lat_raw[1]/60.0 + lat_raw[2]/3600.0
                        elif isinstance(lat_raw, (int, float)):
                            lat = float(lat_raw)
                        else:
                            lat = 0
                            
                        if isinstance(lon_raw, tuple) and len(lon_raw) >= 3:
                            lon = lon_raw[0] + lon_raw[1]/60.0 + lon_raw[2]/3600.0
                        elif isinstance(lon_raw, (int, float)):
                            lon = float(lon_raw)
                        else:
                            lon = 0
                            
                        # 주소로 변환 (행정동)
                        region_name = get_admin_district_from_coords(lat, lon)
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "location": region_name
                        }
                    except Exception as coord_error:
                        logger.warning(f"행정동 변환 오류 (무시됨): {coord_error}")
        except Exception as exif_error:
            logger.warning(f"EXIF data processing error (ignored): {exif_error}")
        
        # GPS 정보가 없으면 기본값 반환
        return {
            "latitude": 37.760028451,  # 파주시청 좌표
            "longitude": 126.779920083,
            "location": "경기도 파주시 시청로 50"
        }
        
    except Exception as e:
        logger.error(f"Location extraction error: {e}")
        return {
            "latitude": 37.760028451,
            "longitude": 126.779920083, 
            "location": "경기도 파주시 시청로 50"
        }


def calculate_urgency(damage_type: str, description: str = "", image_analysis: dict = None, latitude: float = None, longitude: float = None) -> int:
    """긴급 수준 계산 (1-5, 5가 가장 긴급함)"""""
    from advanced_features import emergency_analyzer
    return emergency_analyzer.analyze_emergency_level(damage_type, description, image_analysis, latitude, longitude)


def estimate_processing_time(damage_type: str, urgency_level: int, cluster_info: list = None) -> str:
    """보고서 처리 시간 추정"""
    from advanced_features import time_predictor
    return time_predictor.predict_processing_time(damage_type, urgency_level, cluster_info)


def check_emergency_notification(urgency_level: int, cluster_info: list = None) -> bool:
    """긴급 알림"""
    from advanced_features import notification_system
    return notification_system.should_send_emergency_notification(urgency_level, cluster_info)


def summarize_text_with_textrank(text: str, ratio: float = 0.3) -> str:
    """TextRank 알고리즘을 사용하여 텍스트 요약
    
    Args:
        text: 요약할 텍스트
        ratio: 요약 비율 (0.0 ~ 1.0). 0.3이면 원본의 30% 길이로 요약
    
    Returns:
        요약된 텍스트
    """
    if not text or not text.strip():
        return ""
    
    try:
        from summa import summarizer
        
        # TextRank를 사용한 요약
        summary = summarizer.summarize(text, ratio=ratio, language='korean')
        
        # 요약이 너무 짧거나 비어있으면 원본 텍스트 반환
        if not summary or len(summary.strip()) < 10:
            # 원본이 짧으면 그대로 반환
            if len(text.strip()) < 100:
                return text.strip()
            # 원본이 길면 첫 부분만 반환
            return text.strip()[:200] + "..."
        
        return summary.strip()
        
    except Exception as e:
        logger.warning(f"TextRank 요약 실패, 원본 텍스트 일부 반환: {e}")
        # 요약 실패 시 원본 텍스트의 일부 반환
        if len(text.strip()) < 200:
            return text.strip()
        return text.strip()[:200] + "..."
