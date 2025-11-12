"""API routes and endpoints"""
from fastapi import File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import APIRouter
import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List
from geocoding import get_admin_district_from_coords
from models import ReportRequest, ReportResponse, ReportStatus, StatusUpdate
from database import (
    get_department, get_recent_reports_with_location, 
    get_all_reports, get_statistics, auto_update_status
)
from ai import analyze_image
from utils import extract_location, calculate_urgency, estimate_processing_time, check_emergency_notification, summarize_text_with_textrank
from test_data import create_test_report_data
from cluster import (
    update_map_realtime, perform_dbscan_clustering, 
    analyze_clusters, assign_risk_to_reports, analyze_with_address_input
)
from advanced_features import notification_system
from chat_service import process_query, initialize_chat, is_chat_enabled

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지 (지도 기능 포함)"""
    with open("static/index_with_map.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """관리자 대시보드"""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


from utils import calculate_priority_score
from intersection_data_loader import load_intersection_data
from population_data_loader import load_population_data
from geocoding import get_admin_district_from_coords
import math
import json

# 전역 변수로 교차로 데이터 로드 (서버 시작 시 한 번만)
INTERSECTION_DATA = load_intersection_data()
if INTERSECTION_DATA:
    INTERSECTION_DF = pd.DataFrame(INTERSECTION_DATA)
else:
    INTERSECTION_DF = pd.DataFrame()

# 전역 변수로 인구 가중치 데이터 로드
POPULATION_DF = load_population_data()



def get_population_weight(latitude: float, longitude: float, timestamp: datetime) -> float:
    """신고 지점의 지역과 시간에 맞는 인구 가중치를 반환합니다."""
    logger.info(f"get_population_weight called: lat={latitude}, lon={longitude}, time={timestamp}")
    if not POPULATION_DF or not latitude or not longitude:
        return 0.0

    try:
        region = get_admin_district_from_coords(latitude, longitude)
        logger.info(f"Reverse geocoded address: {region}")
        if not region:
            return 0.0
        
        logger.info(f"Extracted region: {region}")
        if not region:
            return 0.0

        # 시간 키 생성 (예: '09시' -> '09-12시' 범위에 매핑)
        hour = timestamp.hour
        if 0 <= hour < 6:
            time_key = "00-06시"
        elif 6 <= hour < 9:
            time_key = "06-09시"
        elif 9 <= hour < 12:
            time_key = "09-12시"
        elif 12 <= hour < 15:
            time_key = "12-15시"
        elif 15 <= hour < 18:
            time_key = "15-18시"
        elif 18 <= hour < 21:
            time_key = "18-21시"
        else: # 21 <= hour < 24
            time_key = "21-24시"
        
        logger.info(f"Time key: {time_key}")

        # 가중치 조회
        weight = POPULATION_DF.get(region, {}).get(time_key, 0.0)
        logger.info(f"Found weight: {weight}")
        if weight == 0.0:
            logger.warning(f"No population weight found for region: '{region}' at time: '{time_key}'. Returning 0.0.")
        
        return weight

    except Exception as e:
        logger.warning(f"⚠️ Failed to get population weight: {e}")
        return 0.0


def get_nearby_traffic_ratios(latitude: float, longitude: float, radius_km: float = 0.2) -> list:
    """신고 지점 반경 내 교차로의 차량 통행량 비율 목록을 반환합니다."""
    logger.info(f"📍 신고 지점: ({latitude:.6f}, {longitude:.6f})")
    logger.info(f"🚗 검색 반경: {radius_km}km")

    if INTERSECTION_DF.empty:
        logger.error("❌ INTERSECTION_DF가 비어있거나 AVG_TRAFFIC_VOLUME이 0입니다!")
        return []

    logger.info(f"🔍 전체 교차로 수: {len(INTERSECTION_DF)}개")
    
    ratios = []
    found_count = 0
    for _, intersection in INTERSECTION_DF.iterrows():
        # Haversine formula for distance calculation
        dist = math.sqrt(
            ((latitude - intersection['latitude']) * 111.32)**2 +
            ((longitude - intersection['longitude']) * 111.32 * math.cos(math.radians(latitude)))**2
        )
        
        # 약간 더 넓은 범위의 교차로에 대한 디버그 로그
        if dist <= radius_km * 1.5:
            logger.debug(f"교차로 {intersection.get('name', 'N/A')}: 거리={dist:.4f}km")

        if dist <= radius_km:
            ratios.append(intersection['traffic_volume'])
            found_count += 1
    
    logger.info(f"✅ 반경 {radius_km}km 내 교차로: {found_count}개")

    if not ratios:
        logger.warning(f"⚠️ 반경 {radius_km}km 내 교차로가 없습니다. 0 반환.")
        return [0.0]
            
    return ratios

# ... (기존 코드는 생략)

@router.post("/api/report", response_model=ReportResponse)
async def create_report(request: ReportRequest):
    """신고 접수"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        damage_type = request.damage_type or "기타"
        description = request.description or ""
        
        # A: 파손 심각도
        damage_severity = calculate_urgency(damage_type, description)

        # B: 인구 가중치
        population_weight = 0.0
        if request.latitude and request.longitude:
            population_weight = get_population_weight(request.latitude, request.longitude, datetime.now())

        # C: 교차로 차량수 비율
        traffic_ratios = []
        if request.latitude and request.longitude:
            traffic_ratios = get_nearby_traffic_ratios(request.latitude, request.longitude)

        # 최종 우선순위 점수 계산 전 입력값 로깅
        logger.info(f"Calculating urgency with inputs: "
                    f"damage_severity={damage_severity:.2f}, "
                    f"population_weight={population_weight:.4f}, "
                    f"traffic_ratios={traffic_ratios}")

        # 최종 우선순위 점수 계산
        urgency_level = calculate_priority_score(
            damage_severity=damage_severity,
            population_weight=population_weight,
            traffic_ratios=traffic_ratios,
            weights=(0.5, 0.25, 0.25) # (파손 심각도, 인구 가중치, 교통량)
        )
        
        # TextRank 알고리즘을 사용하여 설명 요약 생성
        description_summary = ""
        if description:
            try:
                description_summary = summarize_text_with_textrank(description, ratio=0.3)
                logger.info(f"✅ 설명 요약 생성 완료: 원본 {len(description)}자 → 요약 {len(description_summary)}자")
            except Exception as e:
                logger.warning(f"⚠️ 설명 요약 생성 실패: {e}")
                description_summary = description[:200] + "..." if len(description) > 200 else description
        
        dept_info = get_department(damage_type)
        
        image_path = request.image_path or None
        logger.info(f"신고 접수: user_id={request.user_id}, image_path={image_path}, urgency={urgency_level:.2f}")
        
        cursor.execute('''
        INSERT INTO reports (user_id, damage_type, description, description_summary, urgency_level, status, latitude, longitude, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (request.user_id, damage_type, description, description_summary, urgency_level, '접수', 
              request.latitude, request.longitude, image_path))
        
        report_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # 실시간 지도 업데이트 및 후처리 (기존 로직 유지)
        cluster_info = []
        try:
            reports = get_recent_reports_with_location(days=7)
            if len(reports) >= 1:
                df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=3)
                num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
                
                if num_clusters > 0:
                    cluster_summary = analyze_clusters(df)
                    df = assign_risk_to_reports(df, cluster_summary)
                    
                    for idx, cluster_row in cluster_summary.iterrows():
                        cluster_id = cluster_row['cluster_id']
                        cluster_reports = df[df['cluster'] == cluster_id]
                        addresses = cluster_reports['location'].dropna()
                        if not addresses.empty:
                            cluster_summary.at[idx, 'address'] = addresses.mode()[0] if not addresses.mode().empty else addresses.iloc[0]
                        else:
                            cluster_summary.at[idx, 'address'] = f"위도 {cluster_row['center_lat']:.6f}, 경도 {cluster_row['center_lon']:.6f}"
                        
                        cluster_summary.at[idx, 'max_urgency'] = int(cluster_reports['urgency_level'].max()) if not cluster_reports.empty else 1
                    
                    cluster_info = cluster_summary.to_dict('records')
                
                update_map_realtime(df, cluster_summary if num_clusters > 0 else pd.DataFrame(), 'static/risk_map.html')
                logger.info(f"✅ 실시간 지도 업데이트 완료: {num_clusters}개 군집, {len(df)}개 신고")
        except Exception as cluster_error:
            logger.warning(f"군집 분석 오류 (무시됨): {cluster_error}")

        estimated_time = estimate_processing_time(damage_type, urgency_level, cluster_info)
        should_notify = check_emergency_notification(urgency_level, cluster_info)
        
        response_message = "신고가 성공적으로 접수되었습니다."
        if cluster_info:
            response_message += f" 동일 지역에서 {len(cluster_info)}개의 군집 신고가 탐지되어 우선 처리됩니다."
        if should_notify:
            response_message += " 긴급 신고로 분류되어 즉시 알림이 발송됩니다."
            logger.info(f"긴급 알림 발송: 신고 #{report_id}")
        
        return ReportResponse(
            report_id=report_id,
            status="접수",
            message=response_message,
            damage_type=damage_type,
            urgency_level=urgency_level,
            department=dept_info["department"],
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"신고 접수 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 접수 실패: {str(e)}")


@router.post("/api/upload", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 및 분석"""
    try:
        image_bytes = await file.read()
        
        analysis = analyze_image(image_bytes)
        location_info = extract_location(image_bytes)
        
        os.makedirs("uploads", exist_ok=True)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = f"uploads/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        return {
            "success": True,
            "analysis": analysis,
            "location": location_info,
            "filename": filename,
            "message": "이미지 분석이 완료되었습니다."
        }
        
    except Exception as e:
        logger.error(f"이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")


@router.get("/api/report/{report_id}", response_model=ReportStatus)
async def get_report_status(report_id: int):
    """신고 상태 조회"""
    try:
        auto_update_status()
        
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, created_at, updated_at, damage_type
            FROM reports WHERE id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        status, created_at, updated_at, damage_type = result
        dept_info = get_department(damage_type)
        
        progress_map = {
            '접수': '신고 접수 완료',
            '검토중': '담당 부서 검토 중',
            '처리중': '현장 조사 및 처리 중',
            '완료': '처리 완료'
        }
        
        return ReportStatus(
            report_id=report_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            department=dept_info["department"],
            progress=progress_map.get(status, "처리 중")
        )
        
    except Exception as e:
        logger.error(f"신고 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상태 조회 실패: {str(e)}")


@router.get("/api/damage-types")
async def get_damage_types():
    """손상 유형 목록 조회"""
    return {
        "damage_types": [
            {"id": "가로등", "name": "가로등", "options": ["가로등 꺼짐", "가로등 부서짐", "가로등 깜빡임"]},
            {"id": "도로파손", "name": "도로 파손", "options": ["포장 파손", "싱크홀", "도로 함몰", "차선 불분명"]},
            {"id": "안전펜스", "name": "안전 펜스", "options": ["펜스 파손", "펜스 누락", "펜스 기울어짐"]},
            {"id": "불법주정차", "name": "불법 주정차", "options": ["일반 주정차", "장기 주정차", "위험 주정차"]},
            {"id": "기타", "name": "기타", "options": ["기타 시설물", "기타 안전사고"]}
        ]
    }


@router.get("/api/report/{report_id}/detail")
async def get_report_detail(report_id: int):
    """신고 상세 정보 조회"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_id, damage_type, description, description_summary, urgency_level, status, 
                   created_at, updated_at, latitude, longitude, location, image_path
            FROM reports 
            WHERE id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        report_id, user_id, damage_type, description, description_summary, urgency_level, status, \
        created_at, updated_at, latitude, longitude, location, image_path = result
        
        dept_info = get_department(damage_type)
        urgency_labels = ['낮음', '보통', '높음', '매우높음', '긴급']
        
        return {
            "report_id": report_id,
            "user_id": user_id,
            "damage_type": damage_type,
            "description": description or "설명 없음",
            "description_summary": description_summary or "요약 없음",
            "urgency_level": urgency_level,
            "urgency_label": urgency_labels[round(urgency_level) - 1],
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "latitude": latitude,
            "longitude": longitude,
            "location": location,
            "image_path": image_path,
            "department": dept_info["department"],
            "contact": dept_info["contact"]
        }
        
    except Exception as e:
        logger.error(f"신고 상세 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상세 조회 실패: {str(e)}")


@router.get("/api/reports")
async def get_reports(limit: int = 50, offset: int = 0, status: str = None):
    """신고 목록 조회 (관리자용)"""
    return get_all_reports(limit, offset, status)


@router.get("/api/clusters")
async def get_cluster_reports():
    """군집 신고 현황 조회"""
    try:
        reports = get_recent_reports_with_location(days=7)
        
        if len(reports) < 2:
            return {
                "clusters": [], 
                "total_reports": len(reports),
                "cluster_count": 0,
                "message": "군집 분석을 위한 충분한 데이터가 없습니다. (최소 2개 이상 필요)"
            }
        
        # 데이터베이스에서 반환되는 순서: id, latitude, longitude, urgency_level, created_at, damage_type, location
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'urgency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['emergency_level'] = df['urgency_level']
        
        logger.info(f"데이터프레임 생성 완료: {len(df)}개 신고")
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        logger.info(f"군집 분석 완료: {num_clusters}개 군집 탐지")
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            cluster_info = cluster_summary.to_dict('records')
            
            # DB에서 신고 상세 정보 가져오기
            conn = sqlite3.connect('reports.db')
            cursor = conn.cursor()
            
            for idx, cluster in enumerate(cluster_info):
                cluster_id = cluster['cluster_id']
                cluster_reports = df[df['cluster'] == cluster_id]
                addresses = cluster_reports['location'].dropna()
                if len(addresses) > 0:
                    cluster['address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                else:
                    cluster['address'] = f"위도 {cluster['center_lat']:.6f}, 경도 {cluster['center_lon']:.6f}"
                
                # max_urgency 추가 (advanced_features에서 필요)
                cluster['max_urgency'] = int(cluster_reports['urgency_level'].max()) if len(cluster_reports) > 0 else 1
                
                # 군집에 포함된 신고 상세 정보 가져오기
                report_ids = cluster['report_ids']
                reports_detail = []
                for report_id in report_ids:
                    cursor.execute('''
                        SELECT id, damage_type, description, description_summary, urgency_level, status, created_at, location
                        FROM reports WHERE id = ?
                    ''', (report_id,))
                    result = cursor.fetchone()
                    if result:
                        reports_detail.append({
                            'report_id': result[0],
                            'damage_type': result[1] or '기타',
                            'description': result[2] or '',
                            'description_summary': result[3] or '',
                            'urgency_level': result[4],
                            'status': result[5],
                            'created_at': result[6],
                            'location': result[7] or ''
                        })
                
                cluster['reports'] = reports_detail
                
                # 위험도 근거 생성
                risk_reasons = []
                
                # 긴급도가 높은 신고가 많은 경우
                high_urgency_count = len([r for r in reports_detail if r['urgency_level'] >= 4])
                if high_urgency_count > 0:
                    risk_reasons.append(f"긴급도 높은 신고 {high_urgency_count}건 포함 (긴급도 4 이상)")
                
                # 군집 크기가 큰 경우
                if len(reports_detail) >= 5:
                    risk_reasons.append(f"군집 크기가 큼 ({len(reports_detail)}건)")
                elif len(reports_detail) >= 3:
                    risk_reasons.append(f"군집 크기 적정 ({len(reports_detail)}건)")
                
                # 동일 손상 유형이 많은 경우
                damage_types = [r['damage_type'] for r in reports_detail]
                from collections import Counter
                damage_type_counts = Counter(damage_types)
                most_common_type, count = damage_type_counts.most_common(1)[0]
                if count >= len(reports_detail) * 0.6:  # 60% 이상이 동일 유형
                    risk_reasons.append(f"동일 손상 유형 집중 ({most_common_type} {count}건)")
                
                # 평균 긴급도가 높은 경우
                avg_urgency = cluster['avg_emergency']
                if avg_urgency >= 4.0:
                    risk_reasons.append(f"평균 긴급도 매우 높음 ({avg_urgency:.2f})")
                elif avg_urgency >= 3.0:
                    risk_reasons.append(f"평균 긴급도 높음 ({avg_urgency:.2f})")
                
                cluster['risk_reasons'] = risk_reasons
                
                logger.info(f"군집 {cluster_id} 주소: {cluster['address']}")
            
            conn.close()
            
            logger.info(f"군집 정보 생성 완료: {len(cluster_info)}개")
        else:
            cluster_info = []
            logger.info("군집이 탐지되지 않음")
        
        return {
            "clusters": cluster_info,
            "total_reports": len(reports),
            "cluster_count": num_clusters,
            "message": f"{num_clusters}개의 군집이 탐지되었습니다." if num_clusters > 0 else "군집이 탐지되지 않았습니다."
        }
        
    except Exception as e:
        logger.error(f"군집 신고 조회 오류: {e}")
        return {
            "clusters": [],
            "total_reports": 0,
            "cluster_count": 0,
            "message": f"군집 분석 중 오류가 발생했습니다: {str(e)}"
        }


@router.get("/api/statistics")
async def get_statistics_endpoint():
    """신고 통계 조회"""
    return get_statistics()


@router.post("/api/cluster-analysis")
async def perform_cluster_analysis_endpoint(addresses: List[dict]):
    """주소 기반 군집 분석 수행"""
    try:
        if not addresses or len(addresses) < 2:
            raise HTTPException(status_code=400, detail="최소 2개 이상의 주소가 필요합니다.")
        
        for addr in addresses:
            if 'address' not in addr:
                raise HTTPException(status_code=400, detail="각 주소 정보에 'address' 필드가 필요합니다.")
        
        result = analyze_with_address_input(
            addresses_list=addresses,
            eps_km=0.5,
            min_samples=3,
            output_file='static/risk_map.html'
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="군집 분석에 실패했습니다.")
        
        return {
            "success": True,
            "message": "군집 분석이 완료되었습니다.",
            "clusters": result['clusters'].to_dict('records') if len(result['clusters']) > 0 else [],
            "num_clusters": result['num_clusters'],
            "num_noise": result['num_noise'],
            "map_updated": result['map_updated'],
            "map_url": "/cluster-map"
        }
        
    except Exception as e:
        logger.error(f"군집 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"군집 분석 실패: {str(e)}")


@router.get("/api/update-map")
async def update_cluster_map():
    """실시간 지도 업데이트"""
    try:
        reports = get_recent_reports_with_location(days=7)
        
        if len(reports) < 2:
            return {
                "success": False,
                "message": "지도 업데이트를 위한 충분한 데이터가 없습니다.",
                "map_updated": False
            }
        
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            df = assign_risk_to_reports(df, cluster_summary)
            
            for idx, cluster_row in cluster_summary.iterrows():
                cluster_id = cluster_row['cluster_id']
                cluster_reports = df[df['cluster'] == cluster_id]
                addresses = cluster_reports['location'].dropna()
                if len(addresses) > 0:
                    cluster_summary.at[idx, 'address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                else:
                    cluster_summary.at[idx, 'address'] = f"위도 {cluster_row['center_lat']:.6f}, 경도 {cluster_row['center_lon']:.6f}"
                
                # max_urgency 추가
                cluster_summary.at[idx, 'max_urgency'] = int(cluster_reports['urgency_level'].max()) if len(cluster_reports) > 0 else 1
        else:
            cluster_summary = pd.DataFrame()
        
        update_success = update_map_realtime(df, cluster_summary, 'static/risk_map.html')
        
        return {
            "success": update_success,
            "message": "지도가 성공적으로 업데이트되었습니다." if update_success else "지도 업데이트에 실패했습니다.",
            "num_clusters": num_clusters,
            "total_reports": len(df),
            "map_updated": update_success
        }
        
    except Exception as e:
        logger.error(f"지도 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"지도 업데이트 실패: {str(e)}")


@router.get("/cluster-map", response_class=HTMLResponse)
async def get_cluster_map():
    """군집 분석 지도 페이지"""
    try:
        with open("static/risk_map.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <body>
            <h1>군집 분석 지도가 아직 생성되지 않았습니다.</h1>
            <p>신고가 접수되면 자동으로 생성됩니다.</p>
            <a href="/">메인으로 돌아가기</a>
        </body>
        </html>
        """)


@router.post("/api/report/{report_id}/status")
async def update_report_status_endpoint(report_id: int, data: StatusUpdate):
    """신고 상태 업데이트 (관리자용)"""
    try:
        status = data.status
        valid_statuses = ['접수', '검토중', '처리중', '완료']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 상태입니다. 가능한 상태: {valid_statuses}")
        
        from database import update_report_status
        rowcount = update_report_status(report_id, status)
        
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        return {"message": f"신고 #{report_id}의 상태가 '{status}'로 업데이트되었습니다."}
        
    except Exception as e:
        logger.error(f"신고 상태 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상태 업데이트 실패: {str(e)}")


@router.delete("/api/report/{report_id}")
async def delete_report_endpoint(report_id: int):
    """신고 삭제 (관리자용)"""
    try:
        from database import delete_report
        
        rowcount = delete_report(report_id)
        
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        return {"message": f"신고 #{report_id}가 삭제되었습니다."}
        
    except Exception as e:
        logger.error(f"신고 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 삭제 실패: {str(e)}")


# Test data endpoints
@router.post("/api/test/generate-paju-data")
async def generate_paju_test_data():
    """파주시청 주변 테스트 데이터 생성 (임시 기능)"""
    try:
        test_data = create_test_report_data()
        
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reports (user_id, latitude, longitude, location, damage_type, description, urgency_level, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_data['user_id'],
            test_data['latitude'],
            test_data['longitude'],
            test_data['location'],
            test_data['damage_type'],
            test_data['description'],
            test_data['urgency_level'],
            '접수'
        ))
        
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"테스트 데이터 생성 완료: 신고 #{report_id}")
        
        return {
            "success": True,
            "message": f"파주시청 주변 테스트 데이터가 생성되었습니다. (신고 #{report_id})",
            "report_id": report_id,
            "data": test_data
        }
        
    except Exception as e:
        logger.error(f"테스트 데이터 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 데이터 생성 실패: {str(e)}")


@router.post("/api/test/generate-multiple-paju-data")
async def generate_multiple_paju_test_data(count: int = 5):
    """파주시청 주변 다중 테스트 데이터 생성 (임시 기능)"""
    try:
        import time
        
        if count > 100:
            count = 100
        
        generated_reports = []
        last_api_call = 0
        
        for i in range(count):
            test_data = create_test_report_data()
            
            current_time = time.time()
            if current_time - last_api_call < 1.1:
                time.sleep(1.1 - (current_time - last_api_call))
            last_api_call = time.time()
            
            conn = sqlite3.connect('reports.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO reports (user_id, latitude, longitude, location, damage_type, description, urgency_level, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_data['user_id'],
                test_data['latitude'],
                test_data['longitude'],
                test_data['location'],
                test_data['damage_type'],
                test_data['description'],
                test_data['urgency_level'],
                '접수'
            ))
            
            report_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            generated_reports.append({"report_id": report_id, "data": test_data})
        
        logger.info(f"다중 테스트 데이터 생성 완료: {count}개")
        
        return {
            "success": True,
            "message": f"파주시청 주변 {count}개의 테스트 데이터가 생성되었습니다.",
            "count": count,
            "reports": generated_reports
        }
        
    except Exception as e:
        logger.error(f"다중 테스트 데이터 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"다중 테스트 데이터 생성 실패: {str(e)}")


@router.post("/api/test/generate-100-paju-data")
async def generate_100_paju_test_data():
    """파주시청 주변 100개 테스트 데이터 생성 (임시 기능)"""
    try:
        import time
        
        generated_reports = []
        last_api_call = 0
        
        for i in range(100):
            test_data = create_test_report_data()
            
            current_time = time.time()
            if current_time - last_api_call < 1.1:
                time.sleep(1.1 - (current_time - last_api_call))
            last_api_call = time.time()
            
            conn = sqlite3.connect('reports.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO reports (user_id, latitude, longitude, location, damage_type, description, urgency_level, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_data['user_id'],
                test_data['latitude'],
                test_data['longitude'],
                test_data['location'],
                test_data['damage_type'],
                test_data['description'],
                test_data['urgency_level'],
                '접수'
            ))
            
            report_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            generated_reports.append({"report_id": report_id, "data": test_data})
            
            if (i + 1) % 10 == 0:
                logger.info(f"테스트 데이터 생성 진행: {i + 1}/100")
        
        logger.info(f"100개 테스트 데이터 생성 완료")
        
        return {
            "success": True,
            "message": f"파주시청 주변 100개의 테스트 데이터가 생성되었습니다.",
            "count": 100,
            "reports": generated_reports[:10]
        }
        
    except Exception as e:
        logger.error(f"100개 테스트 데이터 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"100개 테스트 데이터 생성 실패: {str(e)}")


@router.get("/api/test/debug-clusters")
async def debug_clusters():
    """군집 분석 디버깅용 API (임시 기능)"""
    try:
        reports = get_recent_reports_with_location(days=7)
        
        debug_info = {
            "total_reports": len(reports),
            "reports_with_location": len([r for r in reports if r[2] and r[3]]),
            "sample_locations": [
                {"id": r[0], "lat": r[2], "lon": r[3], "type": r[1]} 
                for r in reports[:5]
            ]
        }
        
        if len(reports) >= 2:
            # 데이터베이스에서 반환되는 순서: id, latitude, longitude, urgency_level, created_at, damage_type, location
            df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'urgency_level', 'timestamp', 'damage_type', 'location'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['emergency_level'] = df['urgency_level']
            
            df = perform_dbscan_clustering(df, eps_km=1.0, min_samples=3)
            num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
            
            debug_info.update({
                "clustering_successful": True,
                "num_clusters": num_clusters,
                "cluster_distribution": df['cluster'].value_counts().to_dict(),
                "clusters": df[df['cluster'] != -1].groupby('cluster').agg({
                    'latitude': 'mean',
                    'longitude': 'mean',
                    'report_id': 'count'
                }).to_dict('index') if num_clusters > 0 else {}
            })
        else:
            debug_info["clustering_successful"] = False
            debug_info["error"] = "충분한 데이터가 없습니다"
        
        return debug_info
        
    except Exception as e:
        logger.error(f"군집 디버깅 오류: {e}")
        return {"error": str(e)}


@router.delete("/api/test/delete-all-data")
async def delete_all_test_data():
    """모든 신고 데이터 삭제 (테스트용)"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM reports')
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"모든 신고 데이터 삭제 완료: {deleted_count}개")
        
        return {
            "success": True,
            "message": f"{deleted_count}개의 신고 데이터가 삭제되었습니다.",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"데이터 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 삭제 실패: {str(e)}")


@router.get("/api/map", response_class=HTMLResponse)
async def get_map():
    """군집 분석 지도를 생성하여 HTML로 반환"""
    try:
        from cluster import create_risk_visualization_map
        
        reports = get_recent_reports_with_location(days=7)
        
        if len(reports) < 1:  # 1개 이상이면 지도 표시
            return "<h3>지도를 생성하기에 데이터가 부족합니다.</h3>"
        
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            df = assign_risk_to_reports(df, cluster_summary)
            
            for idx, cluster_row in cluster_summary.iterrows():
                cluster_id = cluster_row['cluster_id']
                cluster_reports = df[df['cluster'] == cluster_id]
                addresses = cluster_reports['location'].dropna()
                if len(addresses) > 0:
                    cluster_summary.at[idx, 'address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                else:
                    cluster_summary.at[idx, 'address'] = f"위도 {cluster_row['center_lat']:.6f}, 경도 {cluster_row['center_lon']:.6f}"
        else:
            cluster_summary = pd.DataFrame()
        
        map_obj = create_risk_visualization_map(df, cluster_summary)
        
        return map_obj.get_root().render()
        
    except Exception as e:
        logger.error(f"지도 생성 오류: {e}")
        return f"<h3>지도 생성 중 오류가 발생했습니다: {e}</h3>"


# ===== 챗봇 엔드포인트 =====
@router.post("/api/chat")
async def chat_endpoint(query: str = Form(...), user_id: str = Form(None)):
    """챗봇 채팅 엔드포인트 (user_id로 신고 내역 조회 가능)"""
    if not is_chat_enabled():
        return JSONResponse({"response": "⚠️ 챗봇 기능이 현재 비활성화되어 있습니다."})
    
    try:
        # user_id가 없으면 None으로 처리 (신고 내역 조회 불가)
        answer = process_query(query, user_id=user_id)
        return JSONResponse({"response": answer})
    except Exception as e:
        logger.error(f"챗봇 오류: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"response": f"⚠️ 서버 오류: {str(e)}"})


@router.get("/api/chat/status")
async def chat_status():
    """챗봇 상태 확인"""
    return {
        "enabled": is_chat_enabled(),
        "message": "챗봇 기능 활성화됨" if is_chat_enabled() else "챗봇 기능 비활성화됨"
    }

