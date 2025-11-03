"""
MCP 클라이언트 애플리케이션
A.X 4.0 VL Light 모델과 MCP 서버를 연결하는 클라이언트
"""
import asyncio
from typing import Optional, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm_handler import LLMHandler
import config


class MCPApp:
    """MCP 클라이언트 애플리케이션"""
    
    def __init__(self):
        """MCP 앱 초기화"""
        print("🚀 MCP 앱 초기화 중...")
        
        # LLM 핸들러 초기화
        self.llm = LLMHandler()
        
        # MCP 세션
        self.session: Optional[ClientSession] = None
        self.available_tools = []
        
        print("✅ MCP 앱 초기화 완료!")
    
    async def connect_to_server(
        self,
        command: str = config.MCP_CONNECTION["command"],
        args: list = config.MCP_CONNECTION["args"]
    ) -> ClientSession:
        """
        MCP 서버에 연결
        
        Args:
            command: 서버 실행 명령어
            args: 명령어 인자
            
        Returns:
            MCP 클라이언트 세션
        """
        print(f"🔌 MCP 서버 연결 중...")
        print(f"   명령어: {command} {' '.join(args)}")
        
        server_params = StdioServerParameters(
            command=command,
            args=args
        )
        
        # stdio를 통한 서버 연결
        read, write = await stdio_client(server_params).__aenter__()
        
        # 세션 생성 및 초기화
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # 사용 가능한 도구 목록 가져오기
        tools_result = await self.session.list_tools()
        self.available_tools = tools_result.tools if hasattr(tools_result, 'tools') else []
        
        print(f"✅ MCP 서버 연결 완료!")
        print(f"   사용 가능한 도구: {len(self.available_tools)}개")
        
        return self.session
    
    async def list_tools(self) -> list:
        """
        사용 가능한 MCP 도구 목록 조회
        
        Returns:
            도구 목록
        """
        if not self.session:
            raise RuntimeError("MCP 서버에 연결되지 않았습니다. connect_to_server()를 먼저 호출하세요.")
        
        tools_result = await self.session.list_tools()
        return tools_result.tools if hasattr(tools_result, 'tools') else []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """
        MCP 도구 호출
        
        Args:
            tool_name: 도구 이름
            arguments: 도구 인자
            
        Returns:
            도구 실행 결과
        """
        if not self.session:
            raise RuntimeError("MCP 서버에 연결되지 않았습니다.")
        
        arguments = arguments or {}
        
        print(f"🔧 도구 호출: {tool_name}")
        result = await self.session.call_tool(tool_name, arguments)
        
        return result
    
    async def process_query(
        self,
        user_input: str,
        use_tools: bool = False
    ) -> str:
        """
        사용자 쿼리 처리
        
        Args:
            user_input: 사용자 입력
            use_tools: MCP 도구 사용 여부
            
        Returns:
            응답 텍스트
        """
        print(f"\n💬 쿼리 처리 중...")
        print(f"   입력: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
        
        # 1. LLM으로 기본 응답 생성
        response = self.llm.generate_response(
            prompt=user_input,
            system_prompt="당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."
        )
        
        # 2. MCP 도구 사용 (옵션)
        if use_tools and self.session:
            # 여기서 LLM 응답을 분석하여 필요한 도구를 호출할 수 있습니다
            # 예: 파일 읽기, 데이터베이스 조회 등
            
            # 사용 가능한 도구 확인
            if self.available_tools:
                print(f"   📦 사용 가능한 도구: {[tool.name for tool in self.available_tools]}")
                # 도구 호출 로직은 나중에 MCP 서버 구현 후 추가
        
        print(f"✅ 응답 생성 완료!")
        return response
    
    async def close(self):
        """MCP 연결 종료"""
        if self.session:
            await self.session.__aexit__(None, None, None)
            print("🔌 MCP 서버 연결 종료")


# 간단한 CLI 인터페이스
async def main():
    """메인 함수 - 간단한 대화형 인터페이스"""
    print("=" * 60)
    print("🤖 A.X 4.0 VL Light MCP 앱")
    print("=" * 60)
    
    # 앱 초기화
    app = MCPApp()
    
    # MCP 서버 연결 (나중에 서버 구현 후 활성화)
    # try:
    #     await app.connect_to_server()
    # except Exception as e:
    #     print(f"⚠️  MCP 서버 연결 실패: {e}")
    #     print("   LLM만 사용하여 계속 진행합니다.")
    
    print("\n💡 사용 방법:")
    print("   - 텍스트 입력 후 Enter")
    print("   - 'quit' 또는 'exit'로 종료")
    print("-" * 60)
    
    try:
        while True:
            # 사용자 입력 받기
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # 종료 명령
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 종료합니다...")
                break
            
            # 쿼리 처리
            try:
                response = await app.process_query(
                    user_input,
                    use_tools=False  # MCP 서버 구현 후 True로 변경
                )
                
                print(f"\n🤖 Assistant: {response}")
                
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
    
    finally:
        # 정리
        await app.close()


if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main())