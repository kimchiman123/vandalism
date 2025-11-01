"""공공기물 파손 신고 챗봇 메인 애플리케이션"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

from database import init_db
from ai import init_yolo_model
from routes import router
from chat_service import initialize_chat

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="공공기물 파손 신고 챗봇", version="1.0.0")

# CORS 설정 (모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# API 라우터 등록
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업 실행"""
    init_db()
    logger.info("✅ 데이터베이스 초기화 완료")
    
    init_yolo_model()
    logger.info("✅ AI 모델 초기화 완료")
    
    # 챗봇 초기화
    if initialize_chat():
        logger.info("✅ 챗봇 초기화 완료")
    else:
        logger.info("⚠️  챗봇 기능이 비활성화되어 있습니다 (API 키 확인 필요)")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
