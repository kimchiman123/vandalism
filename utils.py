"""애플리케이션 유틸리티 함수"""
import logging
from io import BytesIO
from PIL import Image
from geocoding import reverse_geocode
import json
from datetime import datetime
import os
import pandas as pd

# cluster.py에서 행정동 매칭 함수 가져오기
# 순환 참조를 피하기 위해 함수 내에서 import 할 수도 있지만, 구조상 utils는 cluster보다 하위 레벨이므로 직접 import
try:
    from cluster import assign_reports_to_admin_districts
except ImportError:
    # 만약 다른 환경에서 utils만 단독으로 쓰일 경우를 대비
    assign_reports_to_admin_districts = None

logger = logging.getLogger(__name__)

def adjust_urgency_by_population(latitude: float, longitude: float, initial_urgency: int) -> int:
    """
    생활인구 가중치를 적용하여 긴급도를 재계산합니다.

    Args:
        latitude (float): 신고 위도
        longitude (float): 신고 경도
        initial_urgency (int): 초기 긴급도 (1-5)

    Returns:
        int: 조정된 긴급도 (최대 5)
    """
    path = 'data/population_weights.json'
    if not os.path.exists(path):
        logger.warning(f"Population weights file not found at {path}. Skipping adjustment.")
        return initial_urgency

    try:
        with open(path, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        
        location_time_weights = json_file.get('location_time_weights', {})

        # 1. 현재 시간대 파악 
        now = datetime.now()
        current_hour = now.hour
        time_slot_map = {
            range(0, 6): '00-06시',
            range(6, 9): '06-09시',
            range(9, 12): '09-12시',
            range(12, 15): '12-15시',
            range(15, 18): '15-18시',
            range(18, 21): '18-21시',
            range(21, 24): '21-24시'
        }
        current_time_slot = next((slot for r, slot in time_slot_map.items() if current_hour in r), None)

        # 2. 위도/경도를 행정동에 매칭
        if not assign_reports_to_admin_districts:
            logger.warning("assign_reports_to_admin_districts function not available. Skipping adjustment.")
            return initial_urgency
        report_df = pd.DataFrame([{'latitude': latitude, 'longitude': longitude}])
        matched_df = assign_reports_to_admin_districts(report_df)
        
        if matched_df.empty or '행정동명' not in matched_df.columns:
            logger.warning("Could not match coordinates to an administrative district. Skipping adjustment.")
            return initial_urgency
            
        admin_district = matched_df.at[0, '행정동명']

        # 3. 지역-시간대 가중치 조회
        weight = 0
        if admin_district in location_time_weights and current_time_slot:
            weight = location_time_weights[admin_district].get(current_time_slot, 0)
        
        # 최종 긴급도 계산
        scaled_weight = 1 + 4 * min(weight / 2.5, 1)   # 1~5로 변환
        final_urgency = 0.7 * initial_urgency + 0.3 * scaled_weight # 가중 평균
        final_urgency = round(min(final_urgency, 5), 2) # 최댓값 제한 
        
        if final_urgency > initial_urgency:
            logger.info(f"긴급도 조정: 초기 {initial_urgency} -> 최종 {final_urgency})")
        
        return final_urgency

    except Exception as e:
        logger.error(f"Error adjusting urgency with population weights: {e}")
        return initial_urgency


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
                            
                        # 주소로 변환
                        address = reverse_geocode(lat, lon)
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "location": address
                        }
                    except Exception as coord_error:
                        logger.warning(f"GPS coordinate conversion error (ignored): {coord_error}")
        except Exception as exif_error:
            logger.warning(f"EXIF data processing error (ignored): {exif_error}")
        
        # GPS 정보가 없으면 기본값 반환
        return {
            "latitude": 37.5665,  # Seoul City Hall coordinates
            "longitude": 126.9780,
            "location": "서울특별시 중구 세종대로 110"
        }
        
    except Exception as e:
        logger.error(f"Location extraction error: {e}")
        return {
            "latitude": 37.5665,
            "longitude": 126.9780, 
            "location": "서울특별시 중구 세종대로 110"
        }


def calculate_urgency(damage_type: str, description: str = "", image_analysis: dict = None, latitude: float = None, longitude: float = None) -> int:
    """Calculate urgency level (1-5, 5 is most urgent)"""
    from advanced_features import emergency_analyzer
    return emergency_analyzer.analyze_emergency_level(damage_type, description, image_analysis, latitude, longitude)


def estimate_processing_time(damage_type: str, urgency_level: int, cluster_info: list = None) -> str:
    """Estimate processing time for a report"""
    from advanced_features import time_predictor
    return time_predictor.predict_processing_time(damage_type, urgency_level, cluster_info)


def check_emergency_notification(urgency_level: int, cluster_info: list = None) -> bool:
    """Check if emergency notification should be sent"""
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
