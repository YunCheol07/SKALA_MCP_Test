"""
A.X 4.0 Light 모델 핸들러 (텍스트 전용)
안정적인 텍스트 전용 버전 사용
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import config


class LLMHandler:
    """A.X 4.0 Light 모델을 관리하는 핸들러 클래스 (텍스트 전용)"""
    
    def __init__(self, model_name: str = config.MODEL_NAME):
        """
        Args:
            model_name: 허깅페이스 모델 이름
        """
        print(f"🔄 모델 로딩 중: {model_name}")
        print(f"   디바이스: {config.DEVICE}")
        
        self.device = config.DEVICE
        self.model_name = model_name
        
        # 모델 로드
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=config.MODEL_CONFIG["device_map"],
            cache_dir=config.MODEL_CACHE_DIR,
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=config.MODEL_CACHE_DIR,
        )
        
        self.model.eval()
        print(f"✅ 모델 로딩 완료!")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = config.GENERATION_CONFIG["max_new_tokens"],
        temperature: float = config.GENERATION_CONFIG["temperature"],
    ) -> str:
        """
        텍스트 입력으로 응답 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택사항)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            생성된 텍스트 응답
        """
        # 메시지 구성
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 토크나이즈 - tokenize=False로 먼저 텍스트만 생성
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 텍스트를 토큰으로 변환
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        # 생성
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=config.GENERATION_CONFIG["top_p"],
                top_k=config.GENERATION_CONFIG["top_k"],
                do_sample=config.GENERATION_CONFIG["do_sample"],
            )
        
        # 입력 프롬프트 길이 제외
        len_input_prompt = input_ids.shape[1]
        generated_ids = output_ids[0][len_input_prompt:]
        
        # 디코딩
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = config.GENERATION_CONFIG["max_new_tokens"],
    ) -> str:
        """
        대화 형식으로 응답 생성
        
        Args:
            messages: 대화 메시지 리스트
                예: [{"role": "user", "content": "안녕하세요"}]
            max_new_tokens: 생성할 최대 토큰 수
            
        Returns:
            생성된 텍스트 응답
        """
        # 토크나이즈 - tokenize=False로 먼저 텍스트만 생성
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 텍스트를 토큰으로 변환
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=config.GENERATION_CONFIG["temperature"],
                top_p=config.GENERATION_CONFIG["top_p"],
                top_k=config.GENERATION_CONFIG["top_k"],
                do_sample=config.GENERATION_CONFIG["do_sample"],
            )
        
        len_input_prompt = input_ids.shape[1]
        generated_ids = output_ids[0][len_input_prompt:]
        
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return response.strip()


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("A.X 4.0 Light 모델 테스트 (텍스트 전용)")
    print("=" * 50)
    
    # LLM 핸들러 초기화
    llm = LLMHandler()
    
    # 텍스트 전용 테스트
    print("\n📝 텍스트 테스트 1:")
    response = llm.generate_response(
        prompt="인공지능에 대해 간단히 설명해줘.",
        system_prompt="당신은 친절한 AI 어시스턴트입니다."
    )
    print(f"응답: {response}")
    
    print("\n📝 텍스트 테스트 2:")
    response = llm.generate_response(
        prompt="MCP(Model Context Protocol)에 대해 설명해줘.",
    )
    print(f"응답: {response}")
    
    print("\n📝 대화 형식 테스트:")
    messages = [
        {"role": "system", "content": "당신은 파이썬 전문가입니다."},
        {"role": "user", "content": "파이썬에서 리스트와 튜플의 차이점은?"},
    ]
    response = llm.chat(messages)
    print(f"응답: {response}")
    
    print("\n✅ 테스트 완료!")