import pandas as pd
import json
import os
import numpy as np
import logging

# 로거 설정
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
    
    # B: 인구 가중치 스케일링 (cluster.py의 calculate_dynamic_eps와 유사한 동적 스케일링)
    # 로그 스케일을 사용하여 낮은 가중치 구간의 변화를 증폭하고, 높은 가중치 구간의 변화를 완화
    min_weight = 0.0  # 최소 가중치
    max_weight = 2.5  # 최대 가중치 (데이터 분포상 가장 높은 운정3동의 최대값을 기준으로 설정)

    # 로그 스케일 정규화 (0에 가까운 값의 극단적 변화를 막기 위해 0.001을 더함)
    log_pop = np.log(population_weight + 0.001)
    log_min = np.log(min_weight + 0.001)
    log_max = np.log(max_weight + 0.001)

    normalized_pop = (log_pop - log_min) / (log_max - log_min)
    normalized_pop = np.clip(normalized_pop, 0, 1)  # 0~1 사이로 클리핑

    # 1~5 범위로 스케일링
    B_scaled = 1 + 4 * normalized_pop
    
    # C: 교차로 차량수 비율 스케일링
    if traffic_ratios and len(traffic_ratios) > 0:
        # 100m 내 교차로들의 평균 사용
        C_raw = np.mean(traffic_ratios)
        C_capped = min(C_raw, 3.21)
        C_scaled = 1 + 4 * (C_capped / 3.21)
    else:
        # 교차로 데이터 없으면 기본값 1 (최소 위험도)
        C_scaled = 1.0
    
    # 개별 스케일링된 긴급도 로깅
    logger.info(f"Scaled urgency components: Damage(A)={A:.2f}, Population(B)={B_scaled:.2f}, Traffic(C)={C_scaled:.2f}")

    # 최종 가중 평균 계산
    w1, w2, w3 = weights
    final_score = (w1 * A + w2 * B_scaled + w3 * C_scaled)
    
    # 최종 점수가 1~5 범위를 벗어나지 않도록 보정
    return max(1.0, min(final_score, 5.0))


def create_population_weights():
    """
    엑셀 파일에서 유동인구 데이터를 읽어 지역-시간대별 가중치를 계산하고 JSON 파일로 저장합니다.
    """
    file_path = r'C:\Users\baekp\vandalism\open\생활인구분석_경기데이터드림_시간대별분석.xlsx'
    output_path = r'C:\Users\baekp\vandalism\data\population_weights.json'

    if not os.path.exists(file_path):
        print(f"오류: 소스 엑셀 파일을 찾을 수 없습니다 - {file_path}")
        return

    try:
        data = pd.read_excel(file_path)

        data = data.melt(
            id_vars=["행정동명"],            
            var_name="time",              
            value_name="mean_population"    
        )
        data = data.rename(columns={"행정동명":"region"})
        total_population = data['mean_population'].sum()
        data['weight'] = data['mean_population'] / total_population * 100

        location_time_weights = (
            data.pivot(index="region", columns="time", values="weight")
            .to_dict(orient="index")
        )

        final_weights = {"location_time_weights": location_time_weights}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_weights, f, ensure_ascii=False, indent=4)

        """
        # 1. 요일별 데이터 처리
        df_day = pd.read_excel(file_path, sheet_name='29p 요일별 생활인구')
        # 생활인구 컬럼을 min-max scaling을 통해 0~1 사이 값으로 정규화
        min_pop_day = df_day['생활인구'].min()
        max_pop_day = df_day['생활인구'].max()
        df_day['weight'] = (df_day['생활인구'] - min_pop_day) / (max_pop_day - min_pop_day)
        # 요일 이름을 키로 하는 딕셔너리 생성
        day_weights = df_day.set_index('요일')['weight'].to_dict()
        print("요일별 가중치 계산 완료.")

        # 2. 시간대별 데이터 처리
        df_time = pd.read_excel(file_path, sheet_name='35p 시간대별 분석')
        # '행정동명'을 인덱스로 설정
        df_time = df_time.set_index('행정동명')
        
        # 각 행(행정동)별로 시간대 인구를 0~1 사이로 정규화
        df_time_normalized = df_time.apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)
        # NaN 값(모든 시간대 인구가 같을 경우 발생)을 0으로 채움
        df_time_normalized = df_time_normalized.fillna(0)
        # 지역-시간대를 키로 하는 중첩 딕셔너리 생성
        location_time_weights = df_time_normalized.to_dict(orient='index')
        print("지역-시간대별 가중치 계산 완료.")

        # 3. 최종 데이터를 JSON 파일로 저장
        final_weights = {
            'day_weights': day_weights,
            'location_time_weights': location_time_weights
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_weights, f, ensure_ascii=False, indent=4)
        """
        print(f"성공: 가중치 데이터가 {output_path} 에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    create_population_weights()