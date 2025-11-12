import pandas as pd
import json
import os
import numpy as np
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POPULATION_DATA_PATH = os.path.join("open", "생활인구분석_경기데이터드림_시간대별분석.xlsx")

def load_population_data():
    """
    생활인구 수 데이터를 엑셀 파일에서 로드하고, 전처리하여 반환합니다.
    """
    if not os.path.exists(POPULATION_DATA_PATH):
            logger.error(f"생활인구 엑셀 파일을 찾을 수 없습니다: {POPULATION_DATA_PATH}")
            return

    try:
        df = pd.read_excel(POPULATION_DATA_PATH)
        logger.info(f"원본 데이터 컬럼명: {df.columns.tolist()}")

        if "행정동명" not in df.columns:
            logger.error("'행정동명' 컬럼을 찾을 수 없습니다.")
            return

        # 데이터 변환 (시간대별 melt)
        df = df.melt(
            id_vars=["행정동명"],
            var_name="time",
            value_name="mean_population"
        ).rename(columns={"행정동명": "region"})

        # 데이터 정규화 (min-max scaling)
        min_pop = df["mean_population"].min()
        max_pop = df["mean_population"].max()
        if max_pop > min_pop:
            df["weight"] = (df["mean_population"] - min_pop) / (max_pop - min_pop)
        else:
            df["weight"] = 0

        # 피벗 변환 → {region: {time: weight}}
        population = (
            df.pivot(index="region", columns="time", values="weight")
              .fillna(0)
              .to_dict(orient="index")
        )

        logger.info(f"✅ 생활인구 데이터 로드 완료: {len(population)}개")

        return population

    except Exception as e:
        logger.error(f"⚠️ 생활인구 데이터 처리 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    result = load_population_data()
    if result:
        print(f"성공적으로 {len(result)}개 지역의 가중치 데이터를 생성했습니다.")
        print("샘플 데이터:")
        first_region = list(result.items())[0]
        print(first_region)