from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import numpy as np
from transformers import pipeline
import requests
import logging
import pandas as pd
import random
import math
import time
from ultralytics import YOLO

from advanced_features import emergency_analyzer, time_predictor, notification_system
from cluster import (
    update_map_realtime, 
    perform_dbscan_clustering, 
    analyze_clusters, 
    create_risk_visualization_map,
    assign_risk_to_reports,
    analyze_with_address_input
)
from geocoding import geocode_address, reverse_geocode

# ===== 테스트 데이터 생성 함수들 (임시 기능) =====
def generate_random_location_around_paju():
    """파주시청 주변 5km 반지름 내 랜덤 위치 생성"""
    # 파주시청 좌표 (대략)
    paju_center_lat = 37.7597
    paju_center_lon = 126.7775
    
    # 5km 반지름 내 랜덤 위치 생성
    # 1도 위도 ≈ 111km, 1도 경도 ≈ 111km * cos(위도)
    max_lat_offset = 5.0 / 111.0  # 5km를 도 단위로 변환
    max_lon_offset = 5.0 / (111.0 * math.cos(math.radians(paju_center_lat)))
    
    # 균등 분포로 랜덤 오프셋 생성
    lat_offset = random.uniform(-max_lat_offset, max_lat_offset)
    lon_offset = random.uniform(-max_lon_offset, max_lon_offset)
    
    # 실제 거리가 5km 이내인지 확인 (하버사인 공식 사용)
    new_lat = paju_center_lat + lat_offset
    new_lon = paju_center_lon + lon_offset
    
    # 거리 계산
    distance = calculate_distance(paju_center_lat, paju_center_lon, new_lat, new_lon)
    
    # 5km 초과하면 다시 생성
    if distance > 5.0:
        return generate_random_location_around_paju()
    
    return new_lat, new_lon

def calculate_distance(lat1, lon1, lat2, lon2):
    """두 좌표 간의 거리 계산 (km)"""
    R = 6371  # 지구 반지름 (km)
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def generate_random_damage_type():
    """랜덤 신고 유형 생성"""
    damage_types = [
        "가로등",
        "도로파손", 
        "안전펜스",
        "불법주정차",
        "기타"
    ]
    return random.choice(damage_types)

def generate_random_description(damage_type):
    """신고 유형에 따른 랜덤 설명 생성"""
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
    """테스트 신고 데이터 생성"""
    lat, lon = generate_random_location_around_paju()
    damage_type = generate_random_damage_type()
    description = generate_random_description(damage_type)
    
    # 실제 주소 변환 시도 (타임아웃 빠르게)
    try:
        address = reverse_geocode(lat, lon)
        logger.info(f"테스트 데이터 주소 변환 성공: {address}")
    except Exception as e:
        logger.warning(f"테스트 데이터 주소 변환 실패: {e}")
        address = f"경기도 파주시 (위도: {lat:.6f}, 경도: {lon:.6f})"
    
    return {
        "user_id": f"test_user_{random.randint(1000, 9999)}",
        "latitude": lat,
        "longitude": lon,
        "location": address,
        "damage_type": damage_type,
        "description": description,
        "urgency_level": random.randint(1, 5)
    }

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="공공기물 파손 신고 챗봇", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            image_path TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            damage_type TEXT,
            urgency_level INTEGER,
            description TEXT,
            status TEXT DEFAULT '접수',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            damage_type TEXT,
            department_name TEXT,
            contact_info TEXT
        )
    ''')
    
    # 부서 정보 초기 데이터
    departments = [
        ('가로등', '시설관리과', '02-1234-5678'),
        ('도로파손', '도로관리과', '02-1234-5679'),
        ('안전펜스', '교통안전과', '02-1234-5680'),
        ('불법주정차', '교통단속과', '02-1234-5681')
    ]
    
    cursor.execute('DELETE FROM departments')
    cursor.executemany('INSERT INTO departments (damage_type, department_name, contact_info) VALUES (?, ?, ?)', departments)
    
    conn.commit()
    conn.close()

# Pydantic 모델들
class ReportRequest(BaseModel):
    user_id: str
    description: Optional[str] = None
    damage_type: Optional[str] = None

class ReportResponse(BaseModel):
    report_id: int
    status: str
    message: str
    damage_type: str
    urgency_level: int
    department: str
    estimated_time: str

class ReportStatus(BaseModel):
    report_id: int
    status: str
    created_at: str
    updated_at: str
    department: str
    progress: str

# AI 모델 초기화
yolo_model = None
try:
    # YOLO11 튜닝 모델 로드
    logger.info("YOLO11 튜닝 모델을 로딩하는 중...")
    model_path = "yolo_model6/weights/best.pt"
    yolo_model = YOLO(model_path)
    logger.info("✅ YOLO11 튜닝 모델 로드 완료!")
except Exception as e:
    logger.error(f"❌ YOLO11 모델 로드 실패: {e}")
    try:
        # 대안으로 기본 YOLO11 모델 시도
        logger.info("기본 YOLO11 모델 로딩 시도...")
        yolo_model = YOLO('yolo11n.pt')  # 기본 YOLO11 nano 모델
        logger.info("✅ 기본 YOLO11 모델 로드 완료!")
    except Exception as e2:
        logger.error(f"❌ 모든 YOLO 모델 로드 실패: {e2}")
        logger.info("⚠️ AI 기능 없이 기본 모드로 실행합니다.")
        yolo_model = None

# 객체 라벨 한글 번역 함수
def translate_object_label(english_label: str) -> str:
    """영어 객체 라벨을 한글로 번역"""
    translation_map = {
        # 차량 관련
        'car': '자동차',
        'truck': '트럭',
        'bus': '버스',
        'motorcycle': '오토바이',
        'bicycle': '자전거',
        'vehicle': '차량',
        
        # 가로등 관련
        'traffic light': '신호등',
        'pole': '전봇대',
        'lamp': '가로등',
        'street light': '가로등',
        'streetlight': '가로등',
        
        # 도로 관련
        'road': '도로',
        'street': '도로',
        'highway': '고속도로',
        'pavement': '포장도로',
        'asphalt': '아스팔트',
        'concrete': '콘크리트',
        
        # 안전 시설
        'safety_fence': '안전 펜스',
        'barrier': '방호벽',
        'guardrail': '가드레일',
        'railing': '난간',
        
        # 기타 공공시설
        'person': '사람',
        'stop sign': '정지표지판',
        'fire hydrant': '소화전',
        'bench': '벤치',
        'sign': '표지판',
        'building': '건물',
        'tree': '나무',
        'house': '집',
        'window': '창문',
        'door': '문',
        'chair': '의자',
        'table': '테이블',
        'bottle': '병',
        'cup': '컵',
        'book': '책',
        'laptop': '노트북',
        'keyboard': '키보드',
        'mouse': '마우스',
        'tv': 'TV',
        'remote': '리모컨',
        'scissors': '가위'
    }
    
    return translation_map.get(english_label.lower(), english_label)

# 긴급도 판단 함수 (고급 기능 사용)
def calculate_urgency(damage_type: str, description: str = "", image_analysis: dict = None) -> int:
    """긴급도 계산 (1-5, 5가 가장 긴급)"""
    return emergency_analyzer.analyze_emergency_level(damage_type, description, image_analysis)

# 이미지 분석 함수
def analyze_image(image_bytes: bytes) -> dict:
    """이미지에서 객체 탐지 및 분석"""
    if not yolo_model:
        # AI 모델이 없을 때 기본 분석
        try:
            image = Image.open(BytesIO(image_bytes))
            return {
                "damage_type": "기타",
                "confidence": 0.5,
                "detected_objects": [],
                "analysis": "이미지가 성공적으로 업로드되었습니다. 손상 유형을 선택해주세요.",
                "ai_enabled": False
            }
        except Exception as e:
            return {"error": f"이미지 로드 실패: {str(e)}"}
    
    try:
        # 이미지 로드
        image = Image.open(BytesIO(image_bytes))
        
        # YOLO 모델로 객체 탐지
        results = yolo_model(image)
        
        # 탐지할 객체 개수 설정
        MAX_OBJECTS = 5
        
        # 설정된 개수만큼 객체 표시 - 한글 번역 적용
        detected_objects = []
        
        # YOLO 결과 처리
        if results and len(results) > 0:
            result = results[0]  # 첫 번째 결과 사용
            
            # 탐지된 객체들 처리
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                # 신뢰도 순으로 정렬
                sorted_indices = np.argsort(confidences)[::-1]
                
                for i, idx in enumerate(sorted_indices[:MAX_OBJECTS]):
                    confidence = float(confidences[idx])
                    class_id = int(class_ids[idx])
                    
                    # 클래스 이름 가져오기
                    class_name = yolo_model.names[class_id]
                    
                    # 바운딩 박스 좌표
                    box = boxes.xyxy[idx].cpu().numpy()
                    
                    detected_objects.append({
                        'label': translate_object_label(class_name),  # 한글 번역
                        'score': confidence,
                        'box': {
                            'xmin': float(box[0]),
                            'ymin': float(box[1]),
                            'xmax': float(box[2]),
                            'ymax': float(box[3])
                        },
                        'original_label': class_name  # 원본 영어 라벨 보존
                    })
        
        # 공공기물 관련 객체 필터링 및 손상 유형 추정
        public_objects = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 
            'traffic light', 'stop sign', 'fire hydrant', 'bench',
            'pole', 'lamp', 'street light', 'road', 'street', 'highway',
            'safety_fence', 'barrier', 'guardrail', 'railing', 'sign', 'building'
        ]
        
        # 객체에 따른 손상 유형 매핑
        object_to_damage = {
            'car': '불법주정차',
            'truck': '불법주정차', 
            'bus': '불법주정차',
            'motorcycle': '불법주정차',
            'bicycle': '불법주정차',
            'person': '기타',
            'traffic light': '가로등',
            'stop sign': '기타',
            'fire hydrant': '기타',
            'bench': '기타',
            'pole': '가로등',
            'lamp': '가로등',
            'street light': '가로등',
            'road': '도로',
            'street': '도로',
            'highway': '도로',
            'safety_fence': '안전펜스',
            'barrier': '안전펜스',
            'guardrail': '안전펜스',
            'railing': '안전펜스',
            'sign': '기타',
            'building': '기타'
        }
        
        # 탐지된 객체들로 손상 유형 결정
        if detected_objects:
            # 공공기물이 있는지 확인
            public_detected = []
            for obj in detected_objects:
                if obj['original_label'].lower() in public_objects:
                    public_detected.append(obj)
            
            if public_detected:
                # 공공기물이 있으면 가장 높은 신뢰도의 공공기물 사용
                best_object = max(public_detected, key=lambda x: x['score'])
                damage_type = object_to_damage.get(best_object['original_label'].lower(), '기타')
                confidence = best_object['score']
            else:
                # 공공기물이 없으면 가장 높은 신뢰도의 객체 사용
                best_object = detected_objects[0]
                damage_type = "기타"
                confidence = best_object['score']
            
            # 분석 메시지 생성 (여러 객체인 경우)
            if len(detected_objects) == 1:
                analysis = f"탐지된 객체: {detected_objects[0]['label']}"
            else:
                object_names = [obj['label'] for obj in detected_objects]
                analysis = f"탐지된 객체: {', '.join(object_names)}"
        else:
            damage_type = "기타"
            confidence = 0.0
            analysis = "탐지된 객체가 없습니다. 수동으로 손상 유형을 선택해주세요."
        
        return {
            "damage_type": damage_type,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "analysis": analysis,
            "ai_enabled": True
        }
        
    except Exception as e:
        logger.error(f"이미지 분석 오류: {e}")
        return {"error": f"이미지 분석 실패: {str(e)}"}

# 위치 정보 추출 (EXIF 데이터에서)
def extract_location(image_bytes: bytes) -> dict:
    """이미지에서 위치 정보 추출"""
    try:
        image = Image.open(BytesIO(image_bytes))
        
        # EXIF 데이터 안전하게 처리
        try:
            exif = image._getexif()
            if exif:
                # GPS 정보 추출 (실제 구현에서는 더 정교한 GPS 파싱 필요)
                gps_info = exif.get(34853)  # GPS 태그
                if gps_info:
                    # GPS 좌표 처리 (tuple 형태일 수 있음)
                    try:
                        lat_raw = gps_info.get(2)
                        lon_raw = gps_info.get(4)
                        
                        # tuple인 경우 도분초를 도로 변환
                        if isinstance(lat_raw, tuple) and len(lat_raw) >= 3:
                            lat = lat_raw[0] + lat_raw[1]/60.0 + lat_raw[2]/3600.0
                        elif isinstance(lat_raw, (int, float)):
                            lat = float(lat_raw)
                        else:
                            lat = 0
                            
                        if isinstance(lon_raw, tuple) and len(lon_raw) >= 3:
                            lon = lon_raw[0] + lon_raw[1]/60.0 + lon_raw[2]/3600.0
                        elif isinstance(lon_raw, (int, float)):
                            lon = float(lon_raw)
                        else:
                            lon = 0
                            
                        # 주소 변환
                        address = reverse_geocode(lat, lon)
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "location": address
                        }
                    except Exception as coord_error:
                        logger.warning(f"GPS 좌표 변환 오류 (무시됨): {coord_error}")
        except Exception as exif_error:
            logger.warning(f"EXIF 데이터 처리 오류 (무시됨): {exif_error}")
        
        # GPS 정보가 없으면 기본값
        return {
            "latitude": 37.5665,  # 서울시청 좌표
            "longitude": 126.9780,
            "location": "서울특별시 중구 세종대로 110"
        }
        
    except Exception as e:
        logger.error(f"위치 정보 추출 오류: {e}")
        return {
            "latitude": 37.5665,
            "longitude": 126.9780, 
            "location": "서울특별시 중구 세종대로 110"
        }

# 지오코딩 함수는 geocoding.py 모듈로 이동됨

# 부서 연계 함수
def get_department(damage_type: str) -> dict:
    """손상 유형에 따른 담당 부서 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT department_name, contact_info FROM departments WHERE damage_type = ?', (damage_type,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            "department": result[0],
            "contact": result[1]
        }
    else:
        return {
            "department": "시설관리과",
            "contact": "02-1234-5678"
        }

# 처리 예상 시간 계산 (고급 기능 사용)
def estimate_processing_time(damage_type: str, urgency_level: int, cluster_info: List[dict] = None) -> str:
    """처리 예상 시간 계산"""
    return time_predictor.predict_processing_time(damage_type, urgency_level, cluster_info)

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("데이터베이스 초기화 완료")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지 (지도 기능 포함)"""
    with open("static/index_with_map.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """관리자 대시보드"""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# /api/report 엔드포인트 수정 (기존 코드 대체)
@app.post("/api/report", response_model=ReportResponse)
async def create_report(request: ReportRequest):
    """신고 접수"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 기본값 설정
        damage_type = request.damage_type or "기타"
        urgency_level = calculate_urgency(damage_type, request.description or "")
        
        # 부서 정보 조회
        dept_info = get_department(damage_type)
        
        # 신고 데이터 삽입
        cursor.execute('''
        INSERT INTO reports (user_id, damage_type, description, urgency_level, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (request.user_id, damage_type, request.description, urgency_level, '접수'))
        
        report_id = cursor.lastrowid
        
        # 위치 정보 업데이트 (이미지 분석에서 가져온 경우)
        if hasattr(request, 'latitude') and hasattr(request, 'longitude'):
            cursor.execute('''
            UPDATE reports SET latitude = ?, longitude = ? WHERE id = ?
            ''', (request.latitude, request.longitude, report_id))
        
        conn.commit()
        conn.close()
        
        # ===== 실시간 지도 업데이트 (새로 추가) =====
        try:
            # 데이터베이스에서 최근 신고 데이터 로드하여 지도 업데이트
            conn = sqlite3.connect('reports.db')
            cursor = conn.cursor()
            
            recent_time = datetime.now() - timedelta(days=7)
            cursor.execute('''
                SELECT id, latitude, longitude, urgency_level, created_at, damage_type, location
                FROM reports 
                WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY created_at DESC
            ''', (recent_time,))
            
            reports = cursor.fetchall()
            conn.close()
            
            if len(reports) >= 2:
                # DataFrame 생성
                df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 군집 분석 수행 - 300m 반경
                from cluster import perform_dbscan_clustering, analyze_clusters, assign_risk_to_reports
                
                df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
                num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
                
                if num_clusters > 0:
                    cluster_summary = analyze_clusters(df)
                    df = assign_risk_to_reports(df, cluster_summary)
                    
                    # DB 주소를 군집에 추가
                    for idx, cluster_row in cluster_summary.iterrows():
                        cluster_id = cluster_row['cluster_id']
                        cluster_reports = df[df['cluster'] == cluster_id]
                        addresses = cluster_reports['location'].dropna()
                        if len(addresses) > 0:
                            cluster_summary.at[idx, 'address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                        else:
                            cluster_summary.at[idx, 'address'] = f"위도 {cluster_row['center_lat']:.6f}, 경도 {cluster_row['center_lon']:.6f}"
                    
                    cluster_info = cluster_summary.to_dict('records')
                else:
                    cluster_info = []
                
                # 실시간 지도 업데이트
                update_map_realtime(df, cluster_summary if num_clusters > 0 else pd.DataFrame(), 'static/risk_map.html')
                logger.info(f"✅ 실시간 지도 업데이트 완료: {num_clusters}개 군집 탐지")
            else:
                cluster_info = []
                logger.info("⚠️ 지도 업데이트를 위한 충분한 데이터가 없습니다.")
                
        except Exception as cluster_error:
            logger.warning(f"군집 분석 오류 (무시됨): {cluster_error}")
            cluster_info = []
        
        # 처리 예상 시간 계산 (군집 정보 포함)
        estimated_time = estimate_processing_time(damage_type, urgency_level, cluster_info)
        
        # 긴급 알림 발송 여부 확인
        should_notify = notification_system.should_send_emergency_notification(urgency_level, cluster_info)
        
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


@app.post("/api/upload", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 및 분석"""
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        
        # 이미지 분석
        analysis = analyze_image(image_bytes)
        
        # 위치 정보 추출
        location_info = extract_location(image_bytes)
        
        # 이미지 저장
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

# 자동 상태 업데이트 함수
async def auto_update_status():
    """자동으로 신고 상태를 업데이트"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 현재 시간 기준으로 상태 업데이트
        cursor.execute('''
            UPDATE reports 
            SET status = CASE 
                WHEN status = '접수' AND datetime(created_at, '+1 hour') <= datetime('now') THEN '검토중'
                WHEN status = '검토중' AND datetime(created_at, '+4 hours') <= datetime('now') THEN '처리중'
                WHEN status = '처리중' AND datetime(created_at, '+8 hours') <= datetime('now') THEN '완료'
                ELSE status
            END,
            updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('접수', '검토중', '처리중')
        ''')
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"자동 상태 업데이트 오류: {e}")

@app.get("/api/report/{report_id}", response_model=ReportStatus)
async def get_report_status(report_id: int):
    """신고 상태 조회"""
    try:
        # 자동 상태 업데이트 실행
        await auto_update_status()
        
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
        
        # 진행 상황 매핑
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

@app.get("/api/damage-types")
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

@app.get("/api/report/{report_id}/detail")
async def get_report_detail(report_id: int):
    """신고 상세 정보 조회"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_id, damage_type, description, urgency_level, status, 
                   created_at, updated_at, latitude, longitude, location, image_path
            FROM reports 
            WHERE id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        report_id, user_id, damage_type, description, urgency_level, status, \
        created_at, updated_at, latitude, longitude, location, image_path = result
        
        dept_info = get_department(damage_type)
        urgency_labels = ['낮음', '보통', '높음', '매우높음', '긴급']
        
        return {
            "report_id": report_id,
            "user_id": user_id,
            "damage_type": damage_type,
            "description": description or "설명 없음",
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

@app.get("/api/reports")
async def get_reports(limit: int = 50, offset: int = 0, status: str = None):
    """신고 목록 조회 (관리자용)"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 전체 개수 조회
        count_query = 'SELECT COUNT(*) FROM reports'
        count_params = []
        
        if status:
            count_query += ' WHERE status = ?'
            count_params.append(status)
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        # 데이터 조회
        query = '''
            SELECT id, user_id, damage_type, urgency_level, status, created_at, updated_at, latitude, longitude, location
            FROM reports
        '''
        params = []
        
        if status:
            query += ' WHERE status = ?'
            params.append(status)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        reports = cursor.fetchall()
        conn.close()
        
        return {
            "reports": [
                {
                    "id": report[0],
                    "user_id": report[1],
                    "damage_type": report[2],
                    "urgency_level": report[3],
                    "status": report[4],
                    "created_at": report[5],
                    "updated_at": report[6],
                    "latitude": report[7],
                    "longitude": report[8],
                    "location": report[9]
                }
                for report in reports
            ],
            "total": total_count,
            "page": offset // limit + 1 if limit > 0 else 1,
            "per_page": limit,
            "total_pages": (total_count + limit - 1) // limit if limit > 0 else 1
        }
        
    except Exception as e:
        logger.error(f"신고 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 목록 조회 실패: {str(e)}")

@app.get("/api/clusters")
async def get_cluster_reports():
    """군집 신고 현황 조회"""
    try:
        # 최근 7일간의 신고 조회
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        recent_time = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT id, damage_type, latitude, longitude, urgency_level, created_at, location
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        ''', (recent_time,))
        
        reports = cursor.fetchall()
        conn.close()
        
        if len(reports) < 2:
            return {
                "clusters": [], 
                "total_reports": len(reports),
                "cluster_count": 0,
                "message": "군집 분석을 위한 충분한 데이터가 없습니다. (최소 2개 이상 필요)"
            }
        
        # DataFrame 생성 - location 컬럼 포함
        df = pd.DataFrame(reports, columns=['report_id', 'damage_type', 'latitude', 'longitude', 'urgency_level', 'timestamp', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # emergency_level 컬럼 추가 (cluster.py에서 사용)
        df['emergency_level'] = df['urgency_level']
        
        logger.info(f"데이터프레임 생성 완료: {len(df)}개 신고")
        logger.info(f"위치 정보가 있는 신고: {len(df[df['latitude'].notna()])}개")
        
        # 군집 분석 수행 - 300m 반경
        from cluster import perform_dbscan_clustering, analyze_clusters
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        logger.info(f"군집 분석 완료: {num_clusters}개 군집 탐지")
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            cluster_info = cluster_summary.to_dict('records')
            
            # DB에 저장된 주소를 사용하여 군집 중심의 주소 결정 (API 호출 없음)
            for cluster in cluster_info:
                # 군집에 속한 신고들의 주소 가져오기
                cluster_reports = df[df['cluster'] == cluster['cluster_id']]
                
                # 주소가 있는 신고 중 첫 번째 주소 사용
                addresses = cluster_reports['location'].dropna()
                if len(addresses) > 0:
                    # 가장 많이 나타나는 주소 사용 (같은 지역일 가능성 높음)
                    cluster['address'] = addresses.mode()[0] if len(addresses.mode()) > 0 else addresses.iloc[0]
                else:
                    cluster['address'] = f"위도 {cluster['center_lat']:.6f}, 경도 {cluster['center_lon']:.6f}"
                
                logger.info(f"군집 {cluster['cluster_id']} 주소: {cluster['address']}")
            
            logger.info(f"군집 정보 생성 완료: {len(cluster_info)}개 (DB 주소 사용)")
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

@app.get("/api/statistics")
async def get_statistics():
    """신고 통계 조회"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 전체 신고 수
        cursor.execute('SELECT COUNT(*) FROM reports')
        total_reports = cursor.fetchone()[0]
        
        # 긴급도별 신고 수
        cursor.execute('''
            SELECT urgency_level, COUNT(*) 
            FROM reports 
            GROUP BY urgency_level 
            ORDER BY urgency_level
        ''')
        urgency_stats = dict(cursor.fetchall())
        
        # 손상 유형별 신고 수
        cursor.execute('''
            SELECT damage_type, COUNT(*) 
            FROM reports 
            GROUP BY damage_type 
            ORDER BY COUNT(*) DESC
        ''')
        damage_type_stats = dict(cursor.fetchall())
        
        # 최근 24시간 신고 수
        recent_time = datetime.now() - timedelta(hours=24)
        cursor.execute('SELECT COUNT(*) FROM reports WHERE created_at > ?', (recent_time,))
        recent_reports = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_reports": total_reports,
            "recent_24h": recent_reports,
            "urgency_distribution": urgency_stats,
            "damage_type_distribution": damage_type_stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.post("/api/cluster-analysis")
async def perform_cluster_analysis(addresses: List[dict]):
    """주소 기반 군집 분석 수행"""
    try:
        # 주소 리스트 검증
        if not addresses or len(addresses) < 2:
            raise HTTPException(status_code=400, detail="최소 2개 이상의 주소가 필요합니다.")
        
        # 주소 정보 검증
        for addr in addresses:
            if 'address' not in addr:
                raise HTTPException(status_code=400, detail="각 주소 정보에 'address' 필드가 필요합니다.")
        
        # 군집 분석 수행
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

@app.get("/api/update-map")
async def update_cluster_map():
    """실시간 지도 업데이트"""
    try:
        # 데이터베이스에서 최근 신고 데이터 로드
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        recent_time = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT id, latitude, longitude, urgency_level, created_at, damage_type, location
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        ''', (recent_time,))
        
        reports = cursor.fetchall()
        conn.close()
        
        if len(reports) < 2:
            return {
                "success": False,
                "message": "지도 업데이트를 위한 충분한 데이터가 없습니다.",
                "map_updated": False
            }
        
        # DataFrame 생성
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 군집 분석 수행 - 300m 반경
        from cluster import perform_dbscan_clustering, analyze_clusters, assign_risk_to_reports
        
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            df = assign_risk_to_reports(df, cluster_summary)
            
            # DB 주소를 군집에 추가
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
        
        # 실시간 지도 업데이트
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

# 군집 지도 조회 API 추가
@app.get("/cluster-map", response_class=HTMLResponse)
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

class StatusUpdate(BaseModel):
    status: str

@app.post("/api/report/{report_id}/status")
async def update_report_status(report_id: int, data: StatusUpdate):
    """신고 상태 업데이트 (관리자용)"""
    try:
        status = data.status
        valid_statuses = ['접수', '검토중', '처리중', '완료']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 상태입니다. 가능한 상태: {valid_statuses}")
        
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE reports 
            SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (status, report_id))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        conn.commit()
        conn.close()
        
        return {"message": f"신고 #{report_id}의 상태가 '{status}'로 업데이트되었습니다."}
        
    except Exception as e:
        logger.error(f"신고 상태 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상태 업데이트 실패: {str(e)}")

@app.delete("/api/report/{report_id}")
async def delete_report(report_id: int):
    """신고 삭제 (관리자용)"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 신고 존재 확인
        cursor.execute('SELECT id FROM reports WHERE id = ?', (report_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        # 신고 삭제
        cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
        
        conn.commit()
        conn.close()
        
        return {"message": f"신고 #{report_id}가 삭제되었습니다."}
        
    except Exception as e:
        logger.error(f"신고 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 삭제 실패: {str(e)}")

# ===== 테스트 데이터 생성 API (임시 기능) =====
@app.post("/api/test/generate-paju-data")
async def generate_paju_test_data():
    """파주시청 주변 테스트 데이터 생성 (임시 기능)"""
    try:
        # 테스트 데이터 생성
        test_data = create_test_report_data()
        
        # 데이터베이스에 저장
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

@app.post("/api/test/generate-multiple-paju-data")
async def generate_multiple_paju_test_data(count: int = 5):
    """파주시청 주변 다중 테스트 데이터 생성 (임시 기능)"""
    try:
        if count > 100:  # 최대 100개로 제한
            count = 100
        
        generated_reports = []
        
        # API 속도 제한을 위한 타이머
        import time
        last_api_call = 0
        
        for i in range(count):
            # 테스트 데이터 생성
            test_data = create_test_report_data()
            
            # API 속도 제한 (초당 1회)
            current_time = time.time()
            if current_time - last_api_call < 1.1:
                time.sleep(1.1 - (current_time - last_api_call))
            last_api_call = time.time()
            
            # 데이터베이스에 저장
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
            
            generated_reports.append({
                "report_id": report_id,
                "data": test_data
            })
        
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

@app.post("/api/test/generate-100-paju-data")
async def generate_100_paju_test_data():
    """파주시청 주변 100개 테스트 데이터 생성 (임시 기능)"""
    try:
        generated_reports = []
        
        # API 속도 제한을 위한 타이머
        import time
        last_api_call = 0
        
        for i in range(100):
            # 테스트 데이터 생성
            test_data = create_test_report_data()
            
            # API 속도 제한 (초당 1회)
            current_time = time.time()
            if current_time - last_api_call < 1.1:
                time.sleep(1.1 - (current_time - last_api_call))
            last_api_call = time.time()
            
            # 데이터베이스에 저장
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
            
            generated_reports.append({
                "report_id": report_id,
                "data": test_data
            })
            
            # 진행 상황 로그
            if (i + 1) % 10 == 0:
                logger.info(f"테스트 데이터 생성 진행: {i + 1}/100")
        
        logger.info(f"100개 테스트 데이터 생성 완료")
        
        return {
            "success": True,
            "message": f"파주시청 주변 100개의 테스트 데이터가 생성되었습니다.",
            "count": 100,
            "reports": generated_reports[:10]  # 처음 10개만 반환 (성능상)
        }
        
    except Exception as e:
        logger.error(f"100개 테스트 데이터 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"100개 테스트 데이터 생성 실패: {str(e)}")

@app.get("/api/test/debug-clusters")
async def debug_clusters():
    """군집 분석 디버깅용 API (임시 기능)"""
    try:
        # 최근 7일간의 신고 조회
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        recent_time = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT id, damage_type, latitude, longitude, urgency_level, created_at, location
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        ''', (recent_time,))
        
        reports = cursor.fetchall()
        conn.close()
        
        debug_info = {
            "total_reports": len(reports),
            "reports_with_location": len([r for r in reports if r[2] and r[3]]),
            "sample_locations": [
                {"id": r[0], "lat": r[2], "lon": r[3], "type": r[1]} 
                for r in reports[:5]
            ]
        }
        
        if len(reports) >= 2:
            # DataFrame 생성
            df = pd.DataFrame(reports, columns=['report_id', 'damage_type', 'latitude', 'longitude', 'urgency_level', 'timestamp', 'location'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['emergency_level'] = df['urgency_level']
            
            # 군집 분석 수행
            from cluster import perform_dbscan_clustering, analyze_clusters
            
            df = perform_dbscan_clustering(df, eps_km=1.0, min_samples=2)
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

@app.delete("/api/test/delete-all-data")
async def delete_all_test_data():
    """모든 신고 데이터 삭제 (테스트용)"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 모든 신고 데이터 삭제
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

@app.get("/api/map", response_class=HTMLResponse)
async def get_map():
    """
    군집 분석 지도를 생성하여 HTML로 반환
    """
    try:
        # 데이터베이스에서 최근 신고 데이터 로드
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        recent_time = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT id, latitude, longitude, urgency_level, created_at, damage_type, location
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        ''', (recent_time,))
        
        reports = cursor.fetchall()
        conn.close()
        
        if len(reports) < 2:
            return "<h3>지도를 생성하기에 데이터가 부족합니다.</h3>"
        
        # DataFrame 생성
        df = pd.DataFrame(reports, columns=['report_id', 'latitude', 'longitude', 'emergency_level', 'timestamp', 'damage_type', 'location'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 군집 분석 수행 - 300m 반경
        df = perform_dbscan_clustering(df, eps_km=0.3, min_samples=2)
        num_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        
        if num_clusters > 0:
            cluster_summary = analyze_clusters(df)
            from cluster import assign_risk_to_reports
            df = assign_risk_to_reports(df, cluster_summary)
            
            # DB 주소를 군집에 추가
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
        
        # 지도 생성
        map_obj = create_risk_visualization_map(df, cluster_summary)
        
        return map_obj.get_root().render()
        
    except Exception as e:
        logger.error(f"지도 생성 오류: {e}")
        return f"<h3>지도 생성 중 오류가 발생했습니다: {e}</h3>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
