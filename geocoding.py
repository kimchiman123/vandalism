"""
지오코딩 관련 함수들 (주소 ↔ 좌표 변환)
"""
import requests
import logging

logger = logging.getLogger(__name__)


def geocode_address(address: str) -> dict:
    """
    주소를 위도/경도로 변환 (Nominatim API 사용)
    
    Args:
        address: 변환할 주소 문자열
        
    Returns:
        dict: latitude, longitude, address 정보
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "addressdetails": 1,
            "accept-language": "ko",
            "limit": 1
        }
        headers = {
            "User-Agent": "PublicDamageReportBot/1.0"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                return {
                    "latitude": float(result["lat"]),
                    "longitude": float(result["lon"]),
                    "address": result.get("display_name", address)
                }
        
        # API 실패 시 기본값 반환
        return {
            "latitude": 37.5665,
            "longitude": 126.9780,
            "address": address
        }
        
    except Exception as e:
        logger.warning(f"주소 변환 오류: {e}")
        return {
            "latitude": 37.5665,
            "longitude": 126.9780,
            "address": address
        }


def reverse_geocode(latitude: float, longitude: float) -> str:
    """
    위도/경도를 주소로 변환 (Nominatim API 사용)
    
    Args:
        latitude: 위도
        longitude: 경도
        
    Returns:
        str: 변환된 주소 문자열
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "addressdetails": 1,
            "accept-language": "ko"
        }
        headers = {
            "User-Agent": "PublicDamageReportBot/1.0"
        }
        
        # 타임아웃을 1초로 짧게 설정
        response = requests.get(url, params=params, headers=headers, timeout=1)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("address"):
                address = data["address"]
                
                # 한국 주소 형식으로 구성
                country = address.get("country", "")
                state = address.get("state", "")
                city = address.get("city", "")
                town = address.get("town", "")
                village = address.get("village", "")
                suburb = address.get("suburb", "")
                
                # 주소 구성
                if country == "대한민국" or country == "South Korea":
                    if state and city:
                        if town:
                            return f"{state} {city} {town}"
                        elif village:
                            return f"{state} {city} {village}"
                        elif suburb:
                            return f"{state} {city} {suburb}"
                        else:
                            return f"{state} {city}"
                    elif state:
                        return state
                
                # 기본 주소 반환
                display_name = data.get("display_name", "")
                if display_name:
                    # 한국어 주소 부분만 추출
                    parts = display_name.split(", ")
                    korean_parts = [
                        part for part in parts 
                        if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in part)
                    ]
                    if korean_parts:
                        return korean_parts[0]
        
        # API 실패 시 좌표 반환
        return f"위도: {latitude:.6f}, 경도: {longitude:.6f}"
        
    except Exception as e:
        logger.warning(f"주소 변환 오류: {e}")
        return f"위도: {latitude:.6f}, 경도: {longitude:.6f}"

