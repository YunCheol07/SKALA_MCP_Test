"""
MCP 앱 설정 파일
A.X 4.0 VL Light 모델 및 MCP 관련 설정
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# ============================================
# LLM 모델 설정
# ============================================
# VL Light 모델에 호환성 문제가 있어 텍스트 전용 Light 모델 사용
MODEL_NAME = "skt/A.X-4.0-Light"  # 텍스트 전용 (안정적)
# MODEL_NAME = "skt/A.X-4.0-VL-Light"  # VL 모델 (이미지+텍스트, 현재 호환성 문제)
MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # 모델 캐시 디렉토리

# 모델 로딩 설정 (CPU 최적화)
MODEL_CONFIG = {
    "trust_remote_code": True,  # 필수: A.X 모델은 custom code 사용
    "torch_dtype": "float32",  # CPU에서는 float32 사용 (bfloat16은 GPU용)
    "device_map": "cpu",  # CPU 명시적 지정
    "low_cpu_mem_usage": True,  # CPU 메모리 효율적 사용
}

# 생성 파라미터 (CPU 최적화)
GENERATION_CONFIG = {
    "max_new_tokens": 256,  # CPU에서는 토큰 수 줄임 (속도 향상)
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# CPU 최적화 팁
print("💡 CPU 최적화 팁:")
print("   - 응답 생성 시간: 약 30초~1분 예상")
print("   - max_new_tokens를 줄이면 더 빠른 응답 가능")
print("   - 32GB RAM으로 충분히 실행 가능")

# ============================================
# MCP 설정
# ============================================
# Director 사용 여부 (환경 변수로 제어)
USE_DIRECTOR = os.getenv('USE_DIRECTOR', 'false').lower() == 'true'

# MCP 서버 연결 설정
if USE_DIRECTOR:
    MCP_CONNECTION = {
        "command": "director",
        "args": ["serve", "my-proxy"]
    }
else:
    # 직접 MCP 서버 실행 (나중에 서버 구현 후 설정)
    MCP_CONNECTION = {
        "command": "python",
        "args": [str(PROJECT_ROOT / "mcp_server.py")]
    }

# ============================================
# 시스템 설정
# ============================================
# GPU 사용 여부 (내장 그래픽만 있는 경우 False로 설정)
USE_GPU = False  # CPU 전용 모드

# 디바이스 설정
import torch
if USE_GPU and torch.cuda.is_available():
    DEVICE = "cuda"
    print("⚠️  CUDA GPU가 감지되었지만, 내장 그래픽 환경에서는 CPU 사용을 권장합니다.")
else:
    DEVICE = "cpu"
    print("💡 CPU 모드로 실행됩니다. (내장 그래픽 환경)")

# 로그 레벨
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================
# 환경 변수 로드
# ============================================
from dotenv import load_dotenv
load_dotenv()

# 허깅페이스 토큰 (private 모델 사용시 필요)
HF_TOKEN = os.getenv('HF_TOKEN', None)

print(f"✅ 설정 로드 완료")
print(f"📦 모델: {MODEL_NAME}")
print(f"🖥️  디바이스: {DEVICE}")
print(f"🔧 Director 사용: {USE_DIRECTOR}")