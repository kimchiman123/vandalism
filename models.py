"""API 요청 및 응답용 Pydantic 모델"""
from pydantic import BaseModel
from typing import Optional


class ReportRequest(BaseModel):
    """신고 생성 요청 모델"""
    user_id: str
    description: Optional[str] = None
    damage_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ReportResponse(BaseModel):
    """Response model for report creation"""
    report_id: int
    status: str
    message: str
    damage_type: str
    urgency_level: int
    department: str
    estimated_time: str


class ReportStatus(BaseModel):
    """Response model for report status"""
    report_id: int
    status: str
    created_at: str
    updated_at: str
    department: str
    progress: str


class StatusUpdate(BaseModel):
    """Request model for updating report status"""
    status: str

