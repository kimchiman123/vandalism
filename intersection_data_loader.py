
import pandas as pd
import os

INTERSECTION_DATA_PATH = "경기도 파주시_스마트교차로_20250814.csv"

def load_intersection_data():
    """
    교차로 데이터를 CSV 파일에서 로드하고, 필요한 컬럼만 선택하여 반환합니다.
    """
    if not os.path.exists(INTERSECTION_DATA_PATH):
        print(f"오류: 교차로 데이터 파일을 찾을 수 없습니다 - {INTERSECTION_DATA_PATH}")
        return []

    try:
        df = pd.read_csv(INTERSECTION_DATA_PATH, encoding='cp949')
        
        # 필요한 컬럼 선택 및 이름 변경
        df = df[['교차로명', '위도', '경도', '일평균통과차량수']]
        df.columns = ['name', 'latitude', 'longitude', 'traffic_volume']
        
        # 결측치 처리
        df = df.dropna(subset=['latitude', 'longitude', 'traffic_volume'])
        df['traffic_volume'] = pd.to_numeric(df['traffic_volume'], errors='coerce')
        df = df.dropna(subset=['traffic_volume'])

        # 데이터 정규화 (min-max scaling)
        min_traffic = df['traffic_volume'].min()
        max_traffic = df['traffic_volume'].max()
        if max_traffic > min_traffic:
            df['traffic_weight'] = (df['traffic_volume'] - min_traffic) / (max_traffic - min_traffic)
        else:
            df['traffic_weight'] = 0

        return df.to_dict('records')

    except Exception as e:
        print(f"교차로 데이터 로드 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    intersections = load_intersection_data()
    if intersections:
        print(f"성공적으로 {len(intersections)}개의 교차로 데이터를 로드했습니다.")
        print("샘플 데이터:")
        print(intersections[:5])
