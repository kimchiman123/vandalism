import json
from shapely.geometry import shape, Point

# 전역 변수로 경계 데이터 로드
BOUNDARIES = []

def load_boundaries():
    """
    GeoJSON 파일에서 행정구역 경계 데이터를 읽어 전역 변수 BOUNDARIES에 로드합니다.
    서버 시작 시 한 번만 호출되어야 합니다.
    """
    global BOUNDARIES
    try:
        with open('data/paju_submunicipalities.geojson', 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
            # GeoJSON 파일이 FeatureCollection 형태일 경우 'features' 키를 사용
            if geojson_data.get("type") == "FeatureCollection":
                features = geojson_data["features"]
            else:
                # 단일 Feature 혹은 다른 형태의 GeoJSON일 경우를 대비
                features = [geojson_data]
            
            BOUNDARIES = [(feature["properties"]["name"], shape(feature["geometry"])) for feature in features]
            print(f"성공적으로 {len(BOUNDARIES)}개의 행정구역 경계 데이터를 로드했습니다.")
    except FileNotFoundError:
        print("오류: 'data/paju_submunicipalities.geojson' 파일을 찾을 수 없습니다.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"오류: GeoJSON 파일을 파싱하는 중 에러가 발생했습니다: {e}")
    except Exception as e:
        print(f"경계 데이터 로드 중 예기치 않은 오류 발생: {e}")


def get_admin_district_from_coords(lat, lon):
    """
    주어진 위도, 경도 좌표가 속한 행정동명을 반환합니다.

    Args:
        lat (float): 위도
        lon (float): 경도

    Returns:
        str: 행정동명. 어떤 구역에도 속하지 않을 경우 None을 반환합니다.
    """
    if not BOUNDARIES:
        print("경고: 경계 데이터가 로드되지 않았습니다. 먼저 load_boundaries()를 호출해야 합니다.")
        return None
        
    point = Point(lon, lat)  # Shapely는 (X=경도, Y=위도) 순서를 사용합니다.
    for name, polygon in BOUNDARIES:
        if polygon.contains(point):
            return name
    return None

# 이 파일이 직접 실행될 때 테스트 코드를 실행
if __name__ == '__main__':
    # 경계 데이터 로드
    load_boundaries()

    # 테스트 좌표 (예: 파주시청, 운정호수공원, 임진각)
    test_cases = {
        "파주시청": (37.7595, 126.7728),       # 예상: 금촌2동
        "운정호수공원": (37.7193, 126.7712),   # 예상: 운정1동 또는 운정3동
        "임진각": (37.8913, 126.7041),         # 예상: 문산읍
        "헤이리예술마을": (37.7953, 126.6934), # 예상: 탄현면
        "감악산 출렁다리": (37.9060, 126.9840) # 예상: 적성면
    }

    if BOUNDARIES:
        print("\n--- 좌표 기반 행정동 검색 테스트 ---")
        for name, (lat, lon) in test_cases.items():
            district = get_admin_district_from_coords(lat, lon)
            print(f"'{name}' ({lat}, {lon}) -> {district or '알 수 없음'}")

        # GeoJSON 데이터에서 직접 좌표를 추출하여 테스트
        print("\n--- GeoJSON 데이터 기반 랜덤 샘플 테스트 ---")
        if len(BOUNDARIES) > 3:
            import random
            sample_boundaries = random.sample(BOUNDARIES, 3)
            for name, polygon in sample_boundaries:
                # 폴리곤의 중심점을 테스트 좌표로 사용
                centroid = polygon.centroid
                test_lat, test_lon = centroid.y, centroid.x
                district = get_admin_district_from_coords(test_lat, test_lon)
                print(f"'{name}'의 중심점 ({test_lat:.4f}, {test_lon:.4f}) -> {district} (예상: {name})")