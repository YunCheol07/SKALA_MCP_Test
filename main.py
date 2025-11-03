"""
MCP 앱 메인 진입점
"""
import asyncio
import sys
from app import main as app_main


if __name__ == "__main__":
    try:
        asyncio.run(app_main())
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        sys.exit(1)