"""챗봇 서비스 모듈 - RAG (검색 증강 생성) 기반"""
import os
import warnings
from typing import Optional
import logging

# dotenv는 항상 import (챗봇 의존성이 없어도 사용)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# 전역 변수 (초기값)
CHAT_ENABLED = False
genai = None
ConversationBufferMemory = None
HuggingFaceEmbeddings = None
Chroma = None
RecursiveCharacterTextSplitter = None
ChatGoogleGenerativeAI = None


def _load_chat_dependencies():
    """챗봇 의존성 지연 로드"""
    global CHAT_ENABLED, genai, ConversationBufferMemory, HuggingFaceEmbeddings
    global Chroma, RecursiveCharacterTextSplitter, ChatGoogleGenerativeAI
    
    if CHAT_ENABLED:
        return True
    
    try:
        import google.generativeai as genai  # type: ignore
        from langchain.memory import ConversationBufferMemory  # type: ignore
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain_community.vectorstores import Chroma  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        CHAT_ENABLED = True
        return True
    except ImportError as e:
        logging.warning(f"챗봇 의존성 누락: {e}. 챗봇 기능이 비활성화됩니다.")
        CHAT_ENABLED = False
        return False

# 환경 변수 로드
API_KEY = os.getenv("GOOGLE_API_KEY", "")
TXT_PATH = "data/reference.txt"

# 전역 변수
retriever = None
memory = None


def is_chat_enabled() -> bool:
    """챗봇 기능 활성화 여부 확인"""
    return CHAT_ENABLED and API_KEY != ""


def initialize_chat():
    """챗봇 서비스 초기화"""
    global retriever, memory
    
    # Lazy import
    _load_chat_dependencies()
    
    if not is_chat_enabled():
        logger.warning("챗봇 기능이 비활성화되어 있습니다.")
        return False
    
    try:
        # Gemini API 설정
        genai.configure(api_key=API_KEY)
        
        # 문서 로드 및 임베딩
        if os.path.exists(TXT_PATH):
            with open(TXT_PATH, "r", encoding="utf-8") as f:
                text = f.read()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            logger.info("✅ 챗봇 RAG 초기화 완료")
        else:
            logger.warning(f"⚠️ 참조 문서 파일이 없습니다: {TXT_PATH}")
            retriever = None
        
        # 대화 메모리 초기화
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 챗봇 초기화 실패: {e}")
        return False


def build_prompt(context_text: str, query: str) -> str:
    """LLM 프롬프트 생성"""
    return f"""
당신은 '파주시 공공기물 파손 신고 챗봇'입니다.
아래 문서 내용과 이전 대화를 참고하여 시민의 질문에 자연스럽고 명확하게 답하세요.

문서 내용:
{context_text}

사용자 질문:
{query}

응답 지침:
- 질문에 대한 직접적이고 핵심적인 답을 먼저 제공합니다.
- 이전 대화를 확인하고 같은 말을 반복하지 마세요.
- 친근하고 정중한 말투를 사용합니다.
- 신고 내역 조회 관련 질문이면 문서 내용에 포함된 신고 내역 정보를 바탕으로 상세하고 친절하게 답변하세요.
""".strip()


def _is_report_query(query: str) -> bool:
    """신고 내역 조회 관련 질문인지 확인"""
    keywords = [
        "내 신고", "내 기록", "내 접수", "내 신고 내역", "내 신고 목록",
        "제가 신고한", "신고 내역", "신고 조회", "신고 확인",
        "내가 신고한", "접수한 내역", "접수 목록"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def _format_user_reports_for_llm(reports: list) -> str:
    """LLM에게 전달할 텍스트 형식으로 신고 내역 포맷팅"""
    if not reports:
        return "아직 신고 내역이 없습니다."
    
    status_map = {
        "접수": "접수완료",
        "검토중": "검토중",
        "처리중": "처리중",
        "완료": "처리완료"
    }
    
    urgency_map = {
        1: "낮음",
        2: "보통",
        3: "높음",
        4: "매우높음",
        5: "긴급"
    }
    
    text = f"총 {len(reports)}개의 신고 내역이 있습니다:\n\n"
    
    for i, report in enumerate(reports, 1):
        created_at = report["created_at"]
        if isinstance(created_at, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = dt.strftime("%Y년 %m월 %d일 %H:%M")
            except:
                pass
        
        text += f"{i}. 신고번호 #{report['id']}\n"
        text += f"   - 손상유형: {report['damage_type'] or '미지정'}\n"
        text += f"   - 상태: {status_map.get(report['status'], report['status'])}\n"
        text += f"   - 긴급도: {urgency_map.get(report['urgency_level'], '알 수 없음')}\n"
        if report.get('location'):
            text += f"   - 위치: {report['location']}\n"
        if report.get('description_summary'):
            text += f"   - 내용: {report['description_summary']}\n"
        text += f"   - 접수일: {created_at}\n\n"
    
    return text


def process_query(query: str, user_id: Optional[str] = None) -> Optional[str]:
    """RAG + LLM 응답 처리 (user_id로 신고 내역 조회 가능)"""
    # 지연 로드
    _load_chat_dependencies()
    
    if not is_chat_enabled():
        return "⚠️ 챗봇 기능이 현재 비활성화되어 있습니다."
    
    try:
        # 신고 내역 조회 관련 질문인지 확인
        user_reports_data = None
        is_report_query_flag = user_id and _is_report_query(query)
        
        if is_report_query_flag:
            try:
                from database import get_reports_by_user_id
                reports = get_reports_by_user_id(user_id, limit=10)
                user_reports_data = reports  # JSON 데이터로 저장
                user_reports_text = _format_user_reports_for_llm(reports)
                logger.info(f"사용자 {user_id}의 신고 내역 조회: {len(reports)}건")
            except Exception as e:
                logger.error(f"신고 내역 조회 오류: {e}")
                user_reports_text = "신고 내역을 불러오는 중 오류가 발생했습니다."
        
        # 관련 문서 검색
        if retriever:
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = "문서 정보가 없습니다."
        
        # 대화 기록 가져오기
        conversation_history = ""
        if memory and memory.chat_memory.messages:
            conversation_history = "\n".join(
                [f"사용자: {msg.content}" if msg.type == "human" else f"챗봇: {msg.content}"
                 for msg in memory.chat_memory.messages]
            )
        
        # 신고 내역 정보 추가
        if is_report_query_flag and user_reports_text:
            context = f"{user_reports_text}\n\n{context}"
        
        full_context = f"{conversation_history}\n\n{context}"
        prompt = build_prompt(full_context, query)
        
        # LLM 호출 (Gemini API)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY)
        answer = llm.invoke(prompt).content
        
        # HTML 형식으로 변환 (줄바꿈)
        answer = (
            answer.replace(". ", ".<br>")
                  .replace("! ", "!<br>")
                  .replace("? ", "?<br>")
                  .replace("\n", "<br>")
        )
        
        # 신고 내역 조회 질문이면 특별한 마커와 함께 데이터 전달 (LLM 응답 텍스트는 제외)
        if is_report_query_flag and user_reports_data:
            # 특별한 마커를 포함한 응답 (프론트엔드에서 파싱하여 카드 렌더링)
            import json
            reports_json = json.dumps(user_reports_data, ensure_ascii=False, default=str)
            # HTML 이스케이프 처리
            reports_json_escaped = reports_json.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
            # 카드만 반환하고 LLM 응답 텍스트는 포함하지 않음
            answer = f"<div data-report-cards='{reports_json_escaped}'></div>"
        
        # 대화 기록에 저장
        if memory:
            memory.save_context({"input": query}, {"output": answer})
        
        return answer
        
    except Exception as e:
        logger.error(f"챗봇 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ 서버 오류가 발생했습니다: {str(e)}"


def save_chat_context(user_input: str, bot_output: str):
    """대화 기록 저장"""
    if memory:
        memory.save_context({"input": user_input}, {"output": bot_output})

