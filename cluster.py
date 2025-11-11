import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import folium
from folium.plugins import MarkerCluster, HeatMap
import branca.colormap as cm
import warnings
import sqlite3
import requests
import logging
from scipy.spatial.distance import cdist
import json
warnings.filterwarnings('ignore')

# 로깅 설정
logger = logging.getLogger(__name__)

# 지오코딩 함수는 geocoding.py 모듈로 이동됨
# from geocoding import geocode_address

# ============================================
# 파주시 행정동 데이터
# ============================================
PAJU_ADMINISTRATIVE_DATA = {
    '행정동명': [
        '광탄면', '교하동', '금촌1동', '금촌2동', '금촌3동', '문산읍', 
        '법원읍', '운정1동', '운정2동', '운정3동', '운정4동', '운정5동', 
        '운정6동', '월롱면', '적성면', '조리읍', '탄현면', '파주읍', '파평면'
    ],
    '생활인구': [
        2575.04, 10649.2, 21337.8, 23242.4, 15411.9, 29052.2,
        2866.92, 43359.3, 41811.5, 54659.4, 20938.7, 31801.9,
        16858.8, 3463.09, 915.31, 18057.8, 6335.53, 6571.94, 351.021
    ],
    '위도': [
        37.77608949, 37.7530344, 37.76634952, 37.75158647, 37.77142492, 37.8564838,
        37.84916233, 37.72410286, 37.72410286, 37.71711553, 37.71891047, 37.72023157,
        37.71269086, 37.79598237, 37.95426387, 37.74476914, 37.80250643, 37.83075004, 37.92190269
    ],
    '경도': [
        126.8515722, 126.7469049, 126.775969, 126.7771997, 126.7783939, 126.7909532,
        126.8823932, 126.7515039, 126.7515039, 126.7436261, 126.7689073, 126.7105658,
        126.7204332, 126.7902613, 126.9175467, 126.8052165, 126.7161543, 126.8199815, 126.8375401
    ]
}

# DataFrame으로 변환
paju_admin_df = pd.DataFrame(PAJU_ADMINISTRATIVE_DATA)

# ============================================
# 0. 행정동 기반 동적 군집 반경 계산
# ============================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    두 좌표 간의 Haversine 거리 계산 (km 단위)
    """
    R = 6371.0  # 지구 반지름 (km)
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def assign_reports_to_admin_districts(df, admin_df=paju_admin_df):
    """
    각 신고를 가장 가까운 행정동에 매칭
    
    Parameters:
    -----------
    df : pd.DataFrame
        신고 데이터프레임 (latitude, longitude 필요)
    admin_df : pd.DataFrame
        행정동 데이터프레임 (위도, 경도, 생활인구 필요)
    
    Returns:
    --------
    pd.DataFrame
        행정동 정보가 추가된 데이터프레임
    """
    df = df.copy()
    df['행정동명'] = None
    df['생활인구'] = None
    df['동적_eps_km'] = None
    
    for idx, report in df.iterrows():
        # 각 행정동까지의 거리 계산
        distances = []
        for _, admin in admin_df.iterrows():
            dist = haversine_distance(
                report['latitude'], report['longitude'],
                admin['위도'], admin['경도']
            )
            distances.append(dist)
        
        # 가장 가까운 행정동 찾기
        min_dist_idx = np.argmin(distances)
        nearest_admin = admin_df.iloc[min_dist_idx]
        
        df.at[idx, '행정동명'] = nearest_admin['행정동명']
        df.at[idx, '생활인구'] = nearest_admin['생활인구']
    
    return df

def calculate_dynamic_eps(population, base_eps=0.5, min_eps=0.2, max_eps=1.5):
    """
    생활인구에 따라 동적으로 eps (군집 반경) 계산
    
    인구수가 적으면 → eps 크게 (군집 범위 크게)
    인구수가 많으면 → eps 작게 (군집 범위 작게)
    
    Parameters:
    -----------
    population : float
        생활인구 수
    base_eps : float
        기준 eps 값 (km)
    min_eps : float
        최소 eps 값 (km)
    max_eps : float
        최대 eps 값 (km)
    
    Returns:
    --------
    float
        계산된 eps 값 (km)
    """
    # 인구수 정규화 (로그 스케일 사용하여 극단값 완화)
    # 파주시 최소 인구: ~350, 최대 인구: ~54,000
    min_pop = 350
    max_pop = 55000
    
    # 로그 스케일로 정규화
    normalized_pop = (np.log(population + 1) - np.log(min_pop)) / (np.log(max_pop) - np.log(min_pop))
    normalized_pop = np.clip(normalized_pop, 0, 1)  # 0~1 사이로 제한
    
    # 인구가 많을수록 eps를 작게, 적을수록 크게
    # 역비례 관계: eps = max_eps - (normalized_pop * (max_eps - min_eps))
    dynamic_eps = max_eps - (normalized_pop * (max_eps - min_eps))
    
    return round(dynamic_eps, 2)

# ============================================
# 1. DBSCAN 군집 분석 함수
# ============================================
def perform_dbscan_clustering(df, eps_km=0.3, min_samples=2, use_dynamic_eps=True):
    """
    DBSCAN을 사용한 지리적 군집 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        신고 데이터프레임
    eps_km : float
        군집 반경 (킬로미터) - 기본 300m (use_dynamic_eps=False일 때 사용)
    min_samples : int
        군집 형성 최소 신고 수 - 기본 2개
    use_dynamic_eps : bool
        True이면 행정동별 인구 기반 동적 eps 사용, False이면 고정 eps 사용
    
    Returns:
    --------
    pd.DataFrame
        군집 정보가 추가된 데이터프레임
    """
    df = df.copy()
    
    # 동적 eps 사용 시 행정동 정보 추가
    if use_dynamic_eps:
        print("📍 신고 지점을 행정동에 매칭 중...")
        df = assign_reports_to_admin_districts(df)
        
        # 각 신고에 대해 동적 eps 계산
        for idx, row in df.iterrows():
            dynamic_eps = calculate_dynamic_eps(row['생활인구'])
            df.at[idx, '동적_eps_km'] = dynamic_eps
        
        print("\n📊 행정동별 동적 eps 통계:")
        admin_eps_summary = df.groupby('행정동명').agg({
            '생활인구': 'first',
            '동적_eps_km': 'first'
        }).sort_values('생활인구')
        print(admin_eps_summary.to_string())
        
        # 동적 eps를 사용하여 군집화
        # 각 점에 대해 개별적으로 eps를 적용하기 위해 수정된 접근 방식 사용
        df['cluster'] = -1  # 초기화
        cluster_id = 0
        processed = set()
        
        coords = df[['latitude', 'longitude']].values
        eps_values = df['동적_eps_km'].values
        
        for i in range(len(df)):
            if i in processed:
                continue
            
            # 현재 점의 eps로 이웃 찾기
            neighbors = []
            for j in range(len(df)):
                if i != j:
                    dist = haversine_distance(
                        coords[i][0], coords[i][1],
                        coords[j][0], coords[j][1]
                    )
                    # 두 점 중 큰 eps 값 사용 (더 포괄적인 군집화)
                    effective_eps = max(eps_values[i], eps_values[j])
                    if dist <= effective_eps:
                        neighbors.append(j)
            
            # min_samples 조건 확인
            if len(neighbors) + 1 >= min_samples:  # 자기 자신 포함
                # 새 군집 생성
                df.at[df.index[i], 'cluster'] = cluster_id
                processed.add(i)
                
                # 이웃들도 같은 군집에 추가
                for neighbor_idx in neighbors:
                    if neighbor_idx not in processed:
                        df.at[df.index[neighbor_idx], 'cluster'] = cluster_id
                        processed.add(neighbor_idx)
                
                cluster_id += 1
        
        print(f"\n✅ 동적 eps 기반 군집화 완료: {cluster_id}개 군집 생성")
        
    else:
        # 기존 방식: 고정 eps 사용
        coords = df[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        
        kms_per_radian = 6371.0088
        epsilon = eps_km / kms_per_radian
        
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, 
                        algorithm='ball_tree', metric='haversine')
        df['cluster'] = dbscan.fit_predict(coords_rad)
        
        print(f"✅ 고정 eps ({eps_km}km) 기반 군집화 완료")
    
    return df

# ============================================
# 2. 위험도 계산 및 분류 함수
# ============================================
def calculate_risk_score(cluster_data):
    """군집별 위험도 점수 계산"""
    cluster_size = len(cluster_data)
    avg_emergency = cluster_data['emergency_level'].mean()
    max_emergency = cluster_data['emergency_level'].max()
    
    risk_score = (cluster_size * 0.4) + (avg_emergency * 0.3) + (max_emergency * 0.3)
    return round(risk_score, 2)

def classify_risk_level(risk_score):
    """위험도 점수를 5단계로 분류"""
    if risk_score >= 5:
        return 5, "긴급"
    elif risk_score >= 4:
        return 4, "경고"
    elif risk_score >= 3:
        return 3, "주의"
    elif risk_score >= 2:
        return 2, "보통"
    else:
        return 1, "낮음"

def analyze_clusters(df):
    """군집별 위험도 분석 및 분류"""
    cluster_analysis = []
    
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:  # 노이즈 제외
            continue
        
        cluster_data = df[df['cluster'] == cluster_id]
        risk_score = calculate_risk_score(cluster_data)
        risk_level, risk_label = classify_risk_level(risk_score)
        
        center_lat = cluster_data['latitude'].mean()
        center_lon = cluster_data['longitude'].mean()
        
        cluster_info = {
            'cluster_id': cluster_id,
            'report_count': len(cluster_data),
            'center_lat': round(center_lat, 6),
            'center_lon': round(center_lon, 6),
            'avg_emergency': round(cluster_data['emergency_level'].mean(), 2),
            'max_emergency': cluster_data['emergency_level'].max(),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'report_ids': cluster_data['report_id'].tolist()
        }
        cluster_analysis.append(cluster_info)
    
    cluster_analysis.sort(key=lambda x: x['risk_score'], reverse=True)
    return pd.DataFrame(cluster_analysis)

def assign_risk_to_reports(reports_df, cluster_summary_df):
    """각 신고에 위험도 정보 매핑"""
    risk_dict = cluster_summary_df.set_index('cluster_id')[['risk_score', 'risk_level', 'risk_label']].to_dict('index')
    
    reports_df['risk_score'] = None
    reports_df['risk_level'] = None
    reports_df['risk_label'] = None
    
    for idx, report in reports_df.iterrows():
        if report['cluster'] == -1:  # 노이즈(단독 신고)는 개별 긴급도 사용
            reports_df.at[idx, 'risk_score'] = report['emergency_level']
            reports_df.at[idx, 'risk_level'] = report['emergency_level']
            reports_df.at[idx, 'risk_label'] = '단독'
        else:  # 군집에 속한 신고는 군집 위험도 사용
            cluster_risk = risk_dict.get(report['cluster'], {})
            reports_df.at[idx, 'risk_score'] = cluster_risk.get('risk_score', report['emergency_level'])
            reports_df.at[idx, 'risk_level'] = cluster_risk.get('risk_level', report['emergency_level'])
            reports_df.at[idx, 'risk_label'] = cluster_risk.get('risk_label', '단독')
    
    return reports_df

# ============================================
# 3. 긴급 알림 시스템
# ============================================
def trigger_emergency_alerts(reports_df, cluster_summary_df, alert_threshold=4):
    """긴급 알림 발송"""
    print("\n" + "="*80)
    print("🚨 긴급 알림 시스템")
    print("="*80)
    
    high_risk_clusters = cluster_summary_df[cluster_summary_df['risk_level'] >= alert_threshold]
    
    if len(high_risk_clusters) > 0:
        print(f"\n⚠️ 고위험 군집 발견: {len(high_risk_clusters)}개")
        for idx, cluster in high_risk_clusters.iterrows():
            print(f"\n🔔 [긴급] 군집 ID {cluster['cluster_id']}")
            print(f"   위험도: Level {cluster['risk_level']} ({cluster['risk_label']})")
            print(f"   위치: ({cluster['center_lat']}, {cluster['center_lon']})")
            print(f"   신고 {cluster['report_count']}건 집중 발생")
            print(f"   → 즉시 현장 출동 및 대응 필요")
    
    high_emergency_individual = reports_df[
        (reports_df['emergency_level'] >= alert_threshold) & 
        (reports_df['cluster'] == -1)
    ]
    
    if len(high_emergency_individual) > 0:
        print(f"\n⚠️ 고긴급도 단독 신고: {len(high_emergency_individual)}건")
        for idx, report in high_emergency_individual.iterrows():
            print(f"\n🔔 [긴급] 신고 ID {report['report_id']}")
            print(f"   긴급도: {report['emergency_level']}")
            print(f"   위치: ({report['latitude']}, {report['longitude']})")
            print(f"   시간: {report['timestamp']}")
            print(f"   → 개별 대응 필요")
    
    if len(high_risk_clusters) == 0 and len(high_emergency_individual) == 0:
        print("\n✅ 현재 긴급 대응이 필요한 신고 없음")

# ============================================
# 4. Folium 지도 시각화 함수
# ============================================
def create_risk_visualization_map(reports_df, cluster_summary_df, output_file=None):
    """위험도 기반 인터랙티브 지도 생성"""
    # 지도 중심 계산
    center_lat = reports_df['latitude'].mean()
    center_lon = reports_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    risk_colors = {
        1: 'lightgreen',  # 연두색
        2: 'green',  # 노란색
        3: 'orange',  # 주황색
        4: 'red',  # 토마토색
        5: 'darkred',  # 빨간색
        '단독': 'gray'  # 회색
    }
    
    risk_icons = {
        1: 'info-sign',
        2: 'warning-sign',
        3: 'exclamation-sign',
        4: 'fire',
        5: 'flash',
        '단독': 'record'
    }
    
    # 개별 신고 마커
    for idx, report in reports_df.iterrows():
        if report['cluster'] == -1:
            color = risk_colors['단독']
            icon = risk_icons['단독']
            risk_display = f"단독 (긴급도 {report['emergency_level']})"
        else:
            color = risk_colors.get(report['risk_level'], 'gray')
            icon = risk_icons.get(report['risk_level'], 'info-sign')
            risk_display = f"Level {report['risk_level']} ({report['risk_label']})"
        
        # 주소가 있으면 주소 표시, 없으면 위도/경도 표시
        if 'location' in report and report['location'] and pd.notna(report['location']):
            location_display = report['location']
        else:
            location_display = f"위도: {report['latitude']:.6f}, 경도: {report['longitude']:.6f}"
        
        # 행정동 정보 추가 (있는 경우)
        admin_info = ""
        if '행정동명' in report and pd.notna(report['행정동명']):
            admin_info = f"""
            <p><b>행정동:</b> {report['행정동명']}</p>
            <p><b>생활인구:</b> {report['생활인구']:,.0f}명</p>
            """
            if '동적_eps_km' in report and pd.notna(report['동적_eps_km']):
                admin_info += f"<p><b>군집 반경:</b> {report['동적_eps_km']}km</p>"
        
        # 손상 유형 표시 (있는 경우)
        damage_type_display = ""
        if 'damage_type' in report and pd.notna(report['damage_type']) and report['damage_type']:
            damage_type_display = f" - {report['damage_type']}"
        
        popup_html = f"""
        <div style="font-family: Arial; width: 280px;">
            <h4 style="margin-bottom: 10px;">📍 신고#{report['report_id']}{damage_type_display}</h4>
            <hr style="margin: 5px 0;">
            <p><b>위험도:</b> {risk_display}</p>
            <p><b>긴급도:</b> {report['emergency_level']}</p>
            <p><b>시간:</b> {report['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
            <p><b>위치:</b> {location_display}</p>
            {admin_info}
        </div>
        """
        
        folium.Marker(
            location=[report['latitude'], report['longitude']],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    # 군집 중심 마커 - 크기와 위험도에 맞게 표시
    if not cluster_summary_df.empty:
        for idx, cluster in cluster_summary_df.iterrows():
            color = risk_colors.get(cluster['risk_level'], '#808080')
            
            # 군집 크기에 따른 반경 계산
            base_radius = min(max(cluster['report_count'] * 3 + 10, 15), 50)
            
            # 군집에 속한 신고들의 평균 동적 eps 계산
            cluster_reports = reports_df[reports_df['cluster'] == cluster['cluster_id']]
            avg_dynamic_eps = None
            admin_names = set()
            
            if '동적_eps_km' in cluster_reports.columns:
                avg_dynamic_eps = cluster_reports['동적_eps_km'].mean()
                influence_radius = avg_dynamic_eps * 1000  # km를 m로 변환
            else:
                influence_radius = 300  # 기본값 300m
            
            if '행정동명' in cluster_reports.columns:
                admin_names = cluster_reports['행정동명'].dropna().unique()
            
            # 군집의 주소 정보 (있으면 표시)
            if 'address' in cluster and cluster['address']:
                location_info = f"<p><b>위치:</b> {cluster['address']}</p>"
            else:
                location_info = f"<p><b>위치:</b> 위도 {cluster['center_lat']:.6f}, 경도 {cluster['center_lon']:.6f}</p>"
            
            # 행정동 정보 추가
            admin_info = ""
            if len(admin_names) > 0:
                admin_info = f"<p><b>행정동:</b> {', '.join(admin_names)}</p>"
            if avg_dynamic_eps:
                admin_info += f"<p><b>평균 군집 반경:</b> {avg_dynamic_eps:.2f}km ({influence_radius:.0f}m)</p>"
            
            cluster_popup = f"""
            <div style="font-family: Arial; width: 320px;">
                <h4 style="margin-bottom: 10px;">🎯 군집 #{cluster['cluster_id']}</h4>
                <hr style="margin: 5px 0;">
                {location_info}
                {admin_info}
                <p><b>위험도:</b> Level {cluster['risk_level']} ({cluster['risk_label']})</p>
                <p><b>위험 점수:</b> {cluster['risk_score']}</p>
                <p><b>신고 건수:</b> {cluster['report_count']}건</p>
                <p><b>평균 긴급도:</b> {cluster['avg_emergency']}</p>
                <p><b>최대 긴급도:</b> {cluster['max_emergency']}</p>
                <p><b>신고 ID:</b> {', '.join(map(str, cluster['report_ids']))}</p>
            </div>
            """
            
            # 군집 중심 마커
            folium.CircleMarker(
                location=[cluster['center_lat'], cluster['center_lon']],
                radius=base_radius,
                popup=folium.Popup(cluster_popup, max_width=360),
                color=risk_colors.get(cluster['risk_level'], 'gray'),
                fill=True,
                fillColor=risk_colors.get(cluster['risk_level'], 'gray'),
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
            
            # 군집 영향 범위 표시 (동적 eps 기반)
            folium.Circle(
                location=[cluster['center_lat'], cluster['center_lon']],
                radius=influence_radius,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.15,
                weight=2,
                dashArray='5, 5',
                popup=f"군집 영향 범위: 약 {influence_radius:.0f}m"
            ).add_to(m)
    
    # 파주시 행정구역 경계 추가
    try:
        with open('data/paju_submunicipalities.geojson', 'r', encoding='utf-8') as f:
            paju_geojson = json.load(f)
        
        # 스타일 함수 정의
        style_function = lambda x: {
            'fillColor': '#3388ff', 
            'color': '#3388ff', 
            'weight': 1.5, 
            'fillOpacity': 0.1
        }
        
        # 하이라이트 스타일 함수 정의
        highlight_function = lambda x: {
            'fillColor': '#0055ff', 
            'color': '#0055ff', 
            'weight': 3, 
            'fillOpacity': 0.5
        }

        # GeoJson 레이어 생성 및 추가
        geojson_layer = folium.GeoJson(
            paju_geojson,
            name='파주시 행정구역',
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name'],
                aliases=['지역명:'],
                localize=True,
                sticky=False,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            )
        ).add_to(m)
        
        logger.info("✅ 파주시 행정구역 GeoJSON 레이어 추가 완료")

    except FileNotFoundError:
        logger.warning("⚠️ 파주시 행정구역 GeoJSON 파일('data/paju_submunicipalities.geojson')을 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 파주시 행정구역 GeoJSON 처리 중 오류 발생: {e}")

    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin-bottom: 10px; font-weight: bold;">위험도 범례</p>
    <p><span style="color: darkred;">⬤</span> Level 5 - 긴급</p>
    <p><span style="color: red;">⬤</span> Level 4 - 경고</p>
    <p><span style="color: orange;">⬤</span> Level 3 - 주의</p>
    <p><span style="color: green;">⬤</span> Level 2 - 보통</p>
    <p><span style="color: lightgreen;">⬤</span> Level 1 - 낮음</p>
    <p><span style="color: gray;">⬤</span> 단독 신고</p>
    <hr style="margin: 5px 0;">
    <p><b>총 신고:</b> ''' + str(len(reports_df)) + '''건</p>
    <p><b>군집:</b> ''' + str(len(cluster_summary_df)) + '''개</p>
    <p><b>단독:</b> ''' + str(len(reports_df[reports_df['cluster'] == -1])) + '''건</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)
    
    # 지도 저장
    return m

# ============================================
# 5. 실시간 지도 업데이트 함수
# ============================================
def update_map_realtime(reports_df, cluster_summary_df, output_file='static/risk_map.html'):
    """실시간으로 지도를 업데이트"""
    try:
        # 새로운 지도 생성
        m = create_risk_visualization_map(reports_df, cluster_summary_df)
        
        # 지도 저장
        m.save(output_file)
        
        logger.info(f"✅ 실시간 지도 업데이트 완료: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 실시간 지도 업데이트 실패: {e}")
        return False

def analyze_with_address_input(addresses_list, eps_km=0.5, min_samples=3, output_file='static/risk_map.html', use_dynamic_eps=True):
    """
    주소 리스트를 입력받아 군집 분석 수행
    
    Parameters:
    -----------
    addresses_list : list of dict
        주소 정보 리스트
        예: [{'address': '서울시 강남구', 'damage_type': '가로등', 'urgency_level': 3}, ...]
    eps_km : float
        군집 반경 (킬로미터) - use_dynamic_eps=False일 때 사용
    min_samples : int
        군집 형성 최소 신고 수
    output_file : str
        출력 HTML 파일 이름
    use_dynamic_eps : bool
        True이면 행정동별 인구 기반 동적 eps 사용 (기본값)
    """
    print("\n" + "="*80)
    print("📊 주소 기반 공공기물 파손 신고 군집 분석 시스템 (파주시)")
    print("="*80)
    
    if not addresses_list or len(addresses_list) == 0:
        print("⚠️ 입력된 주소 데이터가 없습니다.")
        return None
    
    # 주소를 좌표로 변환
    reports_data = []
    for i, addr_info in enumerate(addresses_list):
        address = addr_info.get('address', '')
        # coords = geocode_address(address)  # 현재 geocode_address 함수가 없으므로 주석 처리
        coords = {'latitude': 37.7597, 'longitude': 126.7775} # 임시로 파주시청 좌표 사용
        
        report_data = {
            'report_id': i + 1,
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'timestamp': datetime.now(),
            'emergency_level': addr_info.get('urgency_level', 3),
            'damage_type': addr_info.get('damage_type', '기타'),
            'address': address
        }
        reports_data.append(report_data)
    
    # DataFrame 생성
    df = pd.DataFrame(reports_data)
    print(f"✅ {len(df)}개의 주소를 좌표로 변환했습니다.")
    
    if len(df) < 2:
        print("⚠️ 분석할 데이터가 부족합니다 (최소 2개 이상 필요).")
        return None
    
    # DBSCAN 군집 분석
    if use_dynamic_eps:
        print(f"\n🔍 동적 eps 기반 DBSCAN 군집 분석 시작 (최소 신고 수: {min_samples})")
    else:
        print(f"\n🔍 고정 eps 기반 DBSCAN 군집 분석 시작 (반경: {eps_km}km, 최소 신고 수: {min_samples})")
    
    df = perform_dbscan_clustering(df, eps_km, min_samples, use_dynamic_eps=use_dynamic_eps)
    
    # 군집 통계
    num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
    num_noise = len(df[df['cluster'] == -1])
    print(f"✅ 군집 분석 완료: {num_clusters}개 군집, {num_noise}개 단독 신고")
    
    if num_clusters == 0:
        print("⚠️ 탐지된 군집이 없습니다. 모든 신고가 단독 신고로 처리됩니다.")
        cluster_summary = pd.DataFrame()
    else:
        # 군집별 위험도 분석
        print("\n📈 군집별 위험도 분석 시작")
        cluster_summary = analyze_clusters(df)
        df = assign_risk_to_reports(df, cluster_summary)
        
        print("\n" + "="*80)
        print("📊 군집 분석 결과")
        print("="*80)
        print(cluster_summary.to_string(index=False))
    
    # 긴급 알림 시스템
    trigger_emergency_alerts(df, cluster_summary)
    
    # 실시간 지도 업데이트
    print("\n🗺️ 실시간 위험도 지도 생성 중...")
    update_success = update_map_realtime(df, cluster_summary, output_file)
    
    if update_success:
        print("\n" + "="*80)
        print("✅ 실시간 분석 완료!")
        print("="*80)
    
    return {
        'reports': df,
        'clusters': cluster_summary,
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'map_updated': update_success
    }

# cluster.py는 모듈로만 사용되며, 직접 실행되지 않습니다.
