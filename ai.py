"""AI/ML 이미지 분석 및 객체 탐지 관련 함수"""
import logging
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# AI 모델 초기화
yolo_model = None


def init_yolo_model():
    """YOLO 모델 초기화"""
    global yolo_model
    try:
        logger.info("Loading YOLO11 tuned model...")
        model_path = "yolo_model6/weights/best.pt"
        yolo_model = YOLO(model_path)
        logger.info("✅ YOLO11 tuned model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ YOLO11 model loading failed: {e}")
        try:
            logger.info("Trying to load default YOLO11 model...")
            yolo_model = YOLO('yolo11n.pt')
            logger.info("✅ Default YOLO11 model loaded successfully!")
        except Exception as e2:
            logger.error(f"❌ All YOLO models failed to load: {e2}")
            logger.info("⚠️ Running in basic mode without AI features.")
            yolo_model = None


def translate_object_label(english_label: str) -> str:
    """영어 객체 라벨을 한글로 번역"""
    translation_map = {
        # Vehicles
        'car': '자동차',
        'truck': '트럭',
        'bus': '버스',
        'motorcycle': '오토바이',
        'bicycle': '자전거',
        'vehicle': '차량',
        
        # Street lights
        'traffic light': '신호등',
        'pole': '전봇대',
        'lamp': '가로등',
        'street light': '가로등',
        'streetlight': '가로등',
        'street_light': '가로등',
        
        # Roads
        'road': '도로',
        'street': '도로',
        'highway': '고속도로',
        'pavement': '포장도로',
        'asphalt': '아스팔트',
        'concrete': '콘크리트',
        
        # Safety facilities
        'safety_fence': '안전 펜스',
        'barrier': '방호벽',
        'guardrail': '가드레일',
        'railing': '난간',
        
        # Other public facilities
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
        'scissors': '가위',
        
        # Damage types
        'damage_road': '도로 파손',
        'pothole': '포트홀',
        'crack': '균열',
        'graffiti': '낙서',
        'vandalism': '기물 파손'
    }
    
    return translation_map.get(english_label.lower(), english_label)


def analyze_image(image_bytes: bytes) -> dict:
    """이미지 객체 탐지 분석"""
    if not yolo_model:
        # Basic analysis when AI model is not available
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
            return {"error": f"Image loading failed: {str(e)}"}
    
    try:
        # Load image
        image = Image.open(BytesIO(image_bytes))
        
        # YOLO object detection
        results = yolo_model(image)
        
        # Maximum number of objects to detect
        MAX_OBJECTS = 5
        
        # Process detected objects with Korean translation
        detected_objects = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                # Sort by confidence
                sorted_indices = np.argsort(confidences)[::-1]
                
                for i, idx in enumerate(sorted_indices[:MAX_OBJECTS]):
                    confidence = float(confidences[idx])
                    class_id = int(class_ids[idx])
                    
                    # Get class name
                    class_name = yolo_model.names[class_id]
                    
                    # Get bounding box coordinates
                    box = boxes.xyxy[idx].cpu().numpy()
                    
                    detected_objects.append({
                        'label': translate_object_label(class_name),
                        'score': confidence,
                        'box': {
                            'xmin': float(box[0]),
                            'ymin': float(box[1]),
                            'xmax': float(box[2]),
                            'ymax': float(box[3])
                        },
                        'original_label': class_name
                    })
        
        # Filter public facility objects and estimate damage type
        public_objects = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 
            'traffic light', 'stop sign', 'fire hydrant', 'bench',
            'pole', 'lamp', 'street light', 'street_light', 'road', 'street', 'highway',
            'safety_fence', 'barrier', 'guardrail', 'railing', 'sign', 'building',
            'damage_road', 'pothole', 'crack', 'graffiti', 'vandalism'
        ]
        
        # Map objects to damage types
        object_to_damage = {
            'car': '불법주정차',
            'truck': '불법주정차', 
            'bus': '불법주정차',
            'motorcycle': '불법주정차',
            'bicycle': '불법주정차',
            'person': '기타',
            'traffic light': '신호등',
            'stop sign': '기타',
            'fire hydrant': '기타',
            'bench': '기타',
            'pole': '가로등',
            'lamp': '가로등',
            'street light': '가로등',
            'street_light': '가로등',
            'road': '도로',
            'street': '도로',
            'highway': '도로',
            'safety_fence': '안전펜스',
            'barrier': '안전펜스',
            'guardrail': '안전펜스',
            'railing': '안전펜스',
            'sign': '기타',
            'building': '기타',
            'damage_road': '도로 파손',
            'pothole': '도로 파손',
            'crack': '도로 파손',
            'graffiti': '기물 파손',
            'vandalism': '기물 파손'
        }
        
        # Determine damage type from detected objects
        if detected_objects:
            public_detected = []
            for obj in detected_objects:
                if obj['original_label'].lower() in public_objects:
                    public_detected.append(obj)
            
            if public_detected:
                best_object = max(public_detected, key=lambda x: x['score'])
                damage_type = object_to_damage.get(best_object['original_label'].lower(), '기타')
                confidence = best_object['score']
            else:
                best_object = detected_objects[0]
                damage_type = "기타"
                confidence = best_object['score']
            
            # Generate analysis message
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
        logger.error(f"Image analysis error: {e}")
        return {"error": f"Image analysis failed: {str(e)}"}

