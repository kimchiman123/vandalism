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
warnings.filterwarnings('ignore')

# 로깅 설정
logger = logging.getLogger(__name__)

# 지오코딩 함수는 geocoding.py 모듈로 이동됨
from geocoding import geocode_address

# ============================================
# 1. DBSCAN 군집 분석 함수
# ============================================
def perform_dbscan_clustering(df, eps_km=0.3, min_samples=2):
    """
    DBSCAN을 사용한 지리적 군집 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        신고 데이터프레임
    eps_km : float
        군집 반경 (킬로미터) - 기본 300m
    min_samples : int
        군집 형성 최소 신고 수 - 기본 2개
    
    Returns:
    --------
    pd.DataFrame
        군집 정보가 추가된 데이터프레임
    """
    coords = df[['latitude', 'longitude']].values
    coords_rad = np.radians(coords)
    
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, 
                    algorithm='ball_tree', metric='haversine')
    df['cluster'] = dbscan.fit_predict(coords_rad)
    
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
def create_risk_visualization_map(reports_df, cluster_summary_df):
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
        1: '#90EE90',  # 연두색
        2: '#FFFF99',  # 노란색
        3: '#FFA500',  # 주황색
        4: '#FF6347',  # 토마토색
        5: '#DC143C',  # 빨간색
        '단독': '#808080'  # 회색
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
            color = risk_colors.get(report['risk_level'], '#808080')
            icon = risk_icons.get(report['risk_level'], 'info-sign')
            risk_display = f"Level {report['risk_level']} ({report['risk_label']})"
        
        # 주소가 있으면 주소 표시, 없으면 위도/경도 표시
        if 'location' in report and report['location'] and pd.notna(report['location']):
            location_display = report['location']
        else:
            location_display = f"위도: {report['latitude']:.6f}, 경도: {report['longitude']:.6f}"
        
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin-bottom: 10px;">📍 신고 #{report['report_id']}</h4>
            <hr style="margin: 5px 0;">
            <p><b>위험도:</b> {risk_display}</p>
            <p><b>긴급도:</b> {report['emergency_level']}</p>
            <p><b>시간:</b> {report['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
            <p><b>위치:</b> {location_display}</p>
        </div>
        """
        
        folium.Marker(
            location=[report['latitude'], report['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red' if report.get('risk_level', 1) >= 4 else 'blue', 
                           icon=icon)
        ).add_to(m)
    
    # 군집 중심 마커 - 크기와 위험도에 맞게 표시
    for idx, cluster in cluster_summary_df.iterrows():
        color = risk_colors.get(cluster['risk_level'], '#808080')
        
        # 군집 크기에 따른 반경 계산
        base_radius = min(max(cluster['report_count'] * 3 + 10, 15), 50)
        
        # 군집의 주소 정보 (있으면 표시)
        if 'address' in cluster and cluster['address']:
            location_info = f"<p><b>위치:</b> {cluster['address']}</p>"
        else:
            location_info = f"<p><b>위치:</b> 위도 {cluster['center_lat']:.6f}, 경도 {cluster['center_lon']:.6f}</p>"
        
        cluster_popup = f"""
        <div style="font-family: Arial; width: 300px;">
            <h4 style="margin-bottom: 10px;">🎯 군집 #{cluster['cluster_id']}</h4>
            <hr style="margin: 5px 0;">
            {location_info}
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
            popup=folium.Popup(cluster_popup, max_width=350),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
        
        # 군집 영향 범위 표시 (300m 기준)
        influence_radius = 300  # 고정 300m
        folium.Circle(
            location=[cluster['center_lat'], cluster['center_lon']],
            radius=influence_radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.15,
            weight=2,
            dashArray='5, 5',
            popup=f"군집 영향 범위: 약 {influence_radius}m"
        ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin-bottom: 10px; font-weight: bold;">위험도 범례</p>
    <p><span style="color: #DC143C;">⬤</span> Level 5 - 긴급</p>
    <p><span style="color: #FF6347;">⬤</span> Level 4 - 경고</p>
    <p><span style="color: #FFA500;">⬤</span> Level 3 - 주의</p>
    <p><span style="color: #FFFF99;">⬤</span> Level 2 - 보통</p>
    <p><span style="color: #90EE90;">⬤</span> Level 1 - 낮음</p>
    <p><span style="color: #808080;">⬤</span> 단독 신고</p>
    <hr style="margin: 5px 0;">
    <p><b>총 신고:</b> ''' + str(len(reports_df)) + '''건</p>
    <p><b>군집:</b> ''' + str(len(cluster_summary_df)) + '''개</p>
    <p><b>단독:</b> ''' + str(len(reports_df[reports_df['cluster'] == -1])) + '''건</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 지도 저장
    return m

# ============================================
# 5. 실시간 지도 업데이트 함수
# ============================================
def update_map_realtime(reports_df, cluster_summary_df, output_file='static/risk_map.html'):
    """실시간으로 지도를 업데이트"""
    try:
        # 기존 지도 파일이 있으면 삭제
        import os
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # 새로운 지도 생성
        m = create_risk_visualization_map(reports_df, cluster_summary_df, output_file)
        
        logger.info(f"✅ 실시간 지도 업데이트 완료: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 실시간 지도 업데이트 실패: {e}")
        return False

def analyze_with_address_input(addresses_list, eps_km=0.5, min_samples=3, output_file='static/risk_map.html'):
    """
    주소 리스트를 입력받아 군집 분석 수행
    
    Parameters:
    -----------
    addresses_list : list of dict
        주소 정보 리스트
        예: [{'address': '서울시 강남구', 'damage_type': '가로등', 'urgency_level': 3}, ...]
    eps_km : float
        군집 반경 (킬로미터)
    min_samples : int
        군집 형성 최소 신고 수
    output_file : str
        출력 HTML 파일 이름
    """
    print("\n" + "="*80)
    print("📊 주소 기반 공공기물 파손 신고 군집 분석 시스템")
    print("="*80)
    
    if not addresses_list or len(addresses_list) == 0:
        print("⚠️ 입력된 주소 데이터가 없습니다.")
        return None
    
    # 주소를 좌표로 변환
    reports_data = []
    for i, addr_info in enumerate(addresses_list):
        address = addr_info.get('address', '')
        coords = geocode_address(address)
        
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
    print(f"\n🔍 DBSCAN 군집 분석 시작 (반경: {eps_km}km, 최소 신고 수: {min_samples})")
    df = perform_dbscan_clustering(df, eps_km, min_samples)
    
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
