"""데이터베이스 초기화 및 쿼리 함수"""
import sqlite3
import logging

logger = logging.getLogger(__name__)


def init_db():
    """데이터베이스 테이블 초기화"""
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
            description_summary TEXT,
            status TEXT DEFAULT '접수',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 기존 테이블에 description_summary 필드가 없으면 추가 (마이그레이션)
    cursor.execute('PRAGMA table_info(reports)')
    columns = [column[1] for column in cursor.fetchall()]
    if 'description_summary' not in columns:
        cursor.execute('ALTER TABLE reports ADD COLUMN description_summary TEXT')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            damage_type TEXT,
            department_name TEXT,
            contact_info TEXT
        )
    ''')
    
    # Department information initialization
    departments = [
        ('가로등', '시설관리과', '02-1234-5678'),
        ('도로파손', '도로관리과', '02-1234-5679'),
        ('안전펜스', '교통안전과', '02-1234-5680'),
        ('불법주정차', '교통단속과', '02-1234-5681')
    ]
    
    cursor.execute('DELETE FROM departments')
    cursor.executemany(
        'INSERT INTO departments (damage_type, department_name, contact_info) VALUES (?, ?, ?)', 
        departments
    )
    
    conn.commit()
    conn.close()


def get_department(damage_type: str) -> dict:
    """손상 유형에 대한 부서 정보 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT department_name, contact_info FROM departments WHERE damage_type = ?', 
        (damage_type,)
    )
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


def get_recent_reports_with_location(days: int = 7):
    """최근 위치 정보가 있는 신고 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    from datetime import datetime, timedelta
    recent_time = datetime.now() - timedelta(days=days)
    
    cursor.execute('''
        SELECT id, latitude, longitude, urgency_level, created_at, damage_type, location
        FROM reports 
        WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY created_at DESC
    ''', (recent_time,))
    
    reports = cursor.fetchall()
    conn.close()
    
    return reports


def get_report_by_id(report_id: int):
    """ID로 신고 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result


def update_report_status(report_id: int, status: str):
    """신고 상태 업데이트"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE reports 
        SET status = ?, updated_at = CURRENT_TIMESTAMP 
        WHERE id = ?
    ''', (status, report_id))
    
    conn.commit()
    rowcount = cursor.rowcount
    conn.close()
    
    return rowcount


def delete_report(report_id: int):
    """신고 삭제"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    
    conn.commit()
    rowcount = cursor.rowcount
    conn.close()
    
    return rowcount


def get_all_reports(limit: int = 50, offset: int = 0, status: str = None):
    """페이지네이션을 사용한 모든 신고 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    # Count query
    count_query = 'SELECT COUNT(*) FROM reports'
    count_params = []
    
    if status:
        count_query += ' WHERE status = ?'
        count_params.append(status)
    
    cursor.execute(count_query, count_params)
    total_count = cursor.fetchone()[0]
    
    # Data query
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


def get_reports_by_user_id(user_id: str, limit: int = 50):
    """user_id로 신고 내역 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, damage_type, urgency_level, status, created_at, updated_at, 
               latitude, longitude, location, description, description_summary
        FROM reports 
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (user_id, limit))
    
    reports = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": report[0],
            "damage_type": report[1],
            "urgency_level": report[2],
            "status": report[3],
            "created_at": report[4],
            "updated_at": report[5],
            "latitude": report[6],
            "longitude": report[7],
            "location": report[8],
            "description": report[9],
            "description_summary": report[10]
        }
        for report in reports
    ]


def get_statistics():
    """신고 통계 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    # Total reports
    cursor.execute('SELECT COUNT(*) FROM reports')
    total_reports = cursor.fetchone()[0]
    
    # Urgency level distribution
    cursor.execute('''
        SELECT ROUND(urgency_level), COUNT(*) 
        FROM reports 
        GROUP BY ROUND(urgency_level) 
        ORDER BY ROUND(urgency_level)
    ''')
    urgency_stats_raw = dict(cursor.fetchall())
    urgency_stats = {int(k): v for k, v in urgency_stats_raw.items() if k is not None}
    
    # Damage type distribution
    cursor.execute('''
        SELECT damage_type, COUNT(*) 
        FROM reports 
        GROUP BY damage_type 
        ORDER BY COUNT(*) DESC
    ''')
    damage_type_stats = dict(cursor.fetchall())
    
    # Recent 24h reports
    from datetime import datetime, timedelta
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


def auto_update_status():
    """시간에 따른 신고 상태 자동 업데이트"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
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
        logger.error(f"Auto status update error: {e}")

