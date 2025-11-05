
import pandas as pd
import json
import os

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
