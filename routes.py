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


@router.post("/api/report", response_model=ReportResponse)
async def create_report(request: ReportRequest):
    """신고 접수"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        damage_type = request.damage_type or "기타"
        description = request.description or ""
        urgency_level = calculate_urgency(damage_type, description)
        
        # TextRank 알고리즘을 사용하여 설명 요약 생성
        description_summary = ""
        if description:
            try:
                description_summary = summarize_text_with_textrank(description, ratio=0.3)
                logger.info(f"✅ 설명 요약 생성 완료: 원본 {len(description)}자 → 요약 {len(description_summary)}자")
            except Exception as e:
                logger.warning(f"⚠️ 설명 요약 생성 실패: {e}")
                # 요약 실패 시 원본 텍스트의 일부 사용
                description_summary = description[:200] + "..." if len(description) > 200 else description
        
        dept_info = get_department(damage_type)
        
        # 위치 정보와 함께 삽입 (description_summary 필드 추가)
        cursor.execute('''
        INSERT INTO reports (user_id, damage_type, description, description_summary, urgency_level, status, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (request.user_id, damage_type, description, description_summary, urgency_level, '접수', 
              request.latitude, request.longitude))
        
        report_id = cursor.lastrowid
        
        # location 필드 업데이트 (주소 변환)
        if request.latitude and request.longitude:
            try:
                from geocoding import reverse_geocode
                address = reverse_geocode(request.latitude, request.longitude)
                cursor.execute('''
                    UPDATE reports SET location = ? WHERE id = ?
                ''', (address, report_id))
            except Exception as e:
                logger.warning(f"주소 변환 실패: {e}")
        
        conn.commit()
        conn.close()
        
        # 실시간 지도 업데이트
        try:
            reports = get_recent_reports_with_location(days=7)
            
            if len(reports) >= 1:  # 1개 이상이면 지도 표시
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
                        if len(addresses) > 0:
                            cluster_summary.at[idx, 'address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                        else:
                            cluster_summary.at[idx, 'address'] = f"위도 {cluster_row['center_lat']:.6f}, 경도 {cluster_row['center_lon']:.6f}"
                        
                        # max_urgency 추가 (advanced_features에서 필요)
                        cluster_summary.at[idx, 'max_urgency'] = int(cluster_reports['urgency_level'].max()) if len(cluster_reports) > 0 else 1
                    
                    cluster_info = cluster_summary.to_dict('records')
                else:
                    cluster_info = []
                
                # 군집이 없어도 단독 신고는 표시
                update_map_realtime(df, cluster_summary if num_clusters > 0 else pd.DataFrame(), 'static/risk_map.html')
                logger.info(f"✅ 실시간 지도 업데이트 완료: {num_clusters}개 군집, {len(df)}개 신고")
            else:
                cluster_info = []
                logger.info("⚠️ 지도 업데이트를 위한 충분한 데이터가 없습니다.")
                
        except Exception as cluster_error:
            logger.warning(f"군집 분석 오류 (무시됨): {cluster_error}")
            cluster_info = []
        
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
            "urgency_label": urgency_labels[urgency_level - 1],
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
        
        if len(reports) < 3:
            return {
                "clusters": [], 
                "total_reports": len(reports),
                "cluster_count": 0,
                "message": "군집 분석을 위한 충분한 데이터가 없습니다. (최소 3개 이상 필요)"
            }
        
        # 데이터베이스에서 반환되는 순서: id, latitude, longitude, urgency_level, created_at, damage_type, location
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'urgency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['emergency_level'] = df['urgency_level']
        
        logger.info(f"데이터프레임 생성 완료: {len(df)}개 신고")
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=3)
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
async def chat_endpoint(query: str = Form(...)):
    """챗봇 채팅 엔드포인트"""
    if not is_chat_enabled():
        return JSONResponse({"response": "⚠️ 챗봇 기능이 현재 비활성화되어 있습니다."})
    
    try:
        answer = process_query(query)
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

