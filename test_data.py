"""테스트 데이터 생성 함수"""
import random
import math
import logging
from geocoding import get_admin_district_from_coords

logger = logging.getLogger(__name__)


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates (km)"""
    R = 6371  # Earth radius (km)
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance


def generate_random_location_around_paju():
    """Generate random location within 5km radius of Paju City Hall"""
    # Paju City Hall coordinates (approximate)
    paju_center_lat = 37.7597
    paju_center_lon = 126.7775
    
    # Generate random location within 5km radius
    # 1 degree latitude ≈ 111km, 1 degree longitude ≈ 111km * cos(latitude)
    max_lat_offset = 5.0 / 111.0  # Convert 5km to degrees
    max_lon_offset = 5.0 / (111.0 * math.cos(math.radians(paju_center_lat)))
    
    # Generate random offset
    lat_offset = random.uniform(-max_lat_offset, max_lat_offset)
    lon_offset = random.uniform(-max_lon_offset, max_lon_offset)
    
    # Check if distance is within 5km (using haversine formula)
    new_lat = paju_center_lat + lat_offset
    new_lon = paju_center_lon + lon_offset
    
    distance = calculate_distance(paju_center_lat, paju_center_lon, new_lat, new_lon)
    
    # If distance exceeds 5km, regenerate
    if distance > 5.0:
        return generate_random_location_around_paju()
    
    return new_lat, new_lon


def generate_random_damage_type():
    """Generate random damage type"""
    damage_types = [
        "가로등",
        "도로파손", 
        "안전펜스",
        "불법주정차",
        "기타"
    ]
    return random.choice(damage_types)


def generate_random_description(damage_type):
    """Generate random description for damage type"""
    descriptions = {
        "가로등": [
            "가로등이 꺼져있어서 밤에 위험합니다",
            "가로등이 깜빡거려서 시야가 불안정합니다",
            "가로등이 부서져서 조명이 안됩니다",
            "가로등이 기울어져서 위험해 보입니다"
        ],
        "도로파손": [
            "도로에 구멍이 생겨서 차량 통행이 위험합니다",
            "포장이 벗겨져서 도로가 울퉁불퉁합니다",
            "싱크홀이 생겨서 교통사고 위험이 있습니다",
            "차선이 지워져서 교통이 혼란스럽습니다"
        ],
        "안전펜스": [
            "안전펜스가 부서져서 위험합니다",
            "펜스가 기울어져서 넘어질 것 같습니다",
            "안전펜스가 누락되어 보행자 안전이 위험합니다",
            "펜스가 녹슬어서 부식되었습니다"
        ],
        "불법주정차": [
            "불법 주정차로 인해 교통이 막힙니다",
            "응급차량 통과가 어려운 상황입니다",
            "장기간 주정차된 차량이 있습니다",
            "위험한 곳에 주정차되어 사고 위험이 있습니다"
        ],
        "기타": [
            "공공시설물이 파손되었습니다",
            "안전사고 위험이 있는 상황입니다",
            "시설물이 부서져서 위험합니다",
            "공공기물이 손상되었습니다"
        ]
    }
    return random.choice(descriptions.get(damage_type, descriptions["기타"]))


def create_test_report_data():
    """Create test report data"""
    lat, lon = generate_random_location_around_paju()
    damage_type = generate_random_damage_type()
    description = generate_random_description(damage_type)
    
    # 행정동으로 변환 시도
    try:
        region_name = get_admin_district_from_coords(lat, lon)
        if region_name:
            address = region_name
            logger.info(f"테스트 데이터 행정동 변환 성공: {address}")
        else:
            address = f"경기도 파주시 (좌표: {lat:.6f}, {lon:.6f})"
            logger.info(f"테스트 데이터 행정동을 찾지 못함. 좌표로 대체: {address}")
    except Exception as e:
        logger.warning(f"테스트 데이터 행정동 변환 실패: {e}")
        address = f"경기도 파주시 (좌표: {lat:.6f}, {lon:.6f})"
    
    return {
        "user_id": f"test_user_{random.randint(1000, 9999)}",
        "latitude": lat,
        "longitude": lon,
        "location": address,
        "damage_type": damage_type,
        "description": description,
        "urgency_level": round(random.uniform(1.0, 5.0), 2)
    }

