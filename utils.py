"""애플리케이션 유틸리티 함수"""
import logging
from io import BytesIO
from PIL import Image
from geocoding import reverse_geocode

logger = logging.getLogger(__name__)


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


def calculate_urgency(damage_type: str, description: str = "", image_analysis: dict = None) -> int:
    """Calculate urgency level (1-5, 5 is most urgent)"""
    from advanced_features import emergency_analyzer
    return emergency_analyzer.analyze_emergency_level(damage_type, description, image_analysis)


def estimate_processing_time(damage_type: str, urgency_level: int, cluster_info: list = None) -> str:
    """Estimate processing time for a report"""
    from advanced_features import time_predictor
    return time_predictor.predict_processing_time(damage_type, urgency_level, cluster_info)


def check_emergency_notification(urgency_level: int, cluster_info: list = None) -> bool:
    """Check if emergency notification should be sent"""
    from advanced_features import notification_system
    return notification_system.should_send_emergency_notification(urgency_level, cluster_info)

