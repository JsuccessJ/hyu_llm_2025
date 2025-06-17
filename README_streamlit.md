# 🌸 향수 추천 LLM Streamlit 앱

> **Graph RAG + Llama 3.1 기반 향수 추천 시스템**

이 프로젝트는 Neo4j 그래프 데이터베이스와 훈련된 Llama 3.1 모델을 사용하여 개인화된 향수 추천 서비스를 제공합니다.

## ✨ 주요 기능

- 🧠 **AI 기반 향수 추천**: 사용자의 선호도에 맞는 개인화된 향수 추천
- 📊 **Graph RAG 검색**: Neo4j를 활용한 고도화된 그래프 검색
- 🎯 **직관적인 UI**: Streamlit 기반의 사용자 친화적 인터페이스
- 🔄 **실시간 추천**: 질문 입력 즉시 추천 결과 제공
- 📱 **반응형 디자인**: 다양한 화면 크기에 최적화

## 🛠️ 시스템 요구사항

### 필수 환경
- Python 3.8+
- Neo4j 데이터베이스
- CUDA 지원 GPU (권장)
- Anaconda/Miniconda

### 가상환경
```bash
conda activate yjllm
```

## 📦 설치 및 설정

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 설정하세요:

```env
# Neo4j 데이터베이스 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# 모델 설정
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TOKEN=your_huggingface_token_here

# 옵션: 추가 설정
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 3. Neo4j 데이터베이스 설정
1. Neo4j Desktop 또는 Neo4j Server 설치
2. 데이터베이스 생성 및 시작
3. 향수 데이터 임포트 (`create_perfume_graph.py` 실행)

## 🚀 실행 방법

### 방법 1: 자동 실행 스크립트 (권장)
```bash
./run_app.sh
```

### 방법 2: 수동 실행
```bash
# 1. 가상환경 활성화
conda activate yjllm

# 2. script 폴더로 이동
cd script

# 3. 연결 테스트
python test.py

# 4. Streamlit 앱 실행
streamlit run app_improved.py
```

## 💡 사용 방법

### 1. 웹 브라우저에서 접속
- 주소: `http://localhost:8501`
- 앱이 자동으로 열립니다

### 2. 시스템 상태 확인
- 사이드바에서 시스템 온라인/오프라인 상태 확인
- 🟢 온라인: 모든 기능 사용 가능
- 🔴 오프라인: 연결 문제 해결 필요

### 3. 향수 추천 받기

#### 직접 질문하기
- 메인 입력창에 원하는 향수 조건 입력
- 예시: "20대 여성을 위한 봄에 어울리는 플라워 향수 추천해줘"

#### 샘플 질문 이용하기
- 미리 준비된 샘플 질문 버튼 클릭
- 다양한 상황별 질문 제공

### 4. 효과적인 질문 팁
- **구체적인 정보 포함**: 성별, 연령대, 계절, 상황
- **향조 언급**: 플라워, 우디, 프레시, 오리엔탈 등
- **사용 목적 명시**: 데이트용, 직장용, 일상용 등

## 📋 샘플 질문 예시

### 기본 추천
- "여자 플라워 향수 추천해줘"
- "우디 계열의 남성 향수 뭐가 좋아?"
- "가을에 어울리는 향수 알려줘"

### 상황별 추천
- "데이트할 때 좋은 로맨틱한 향수"
- "직장에서 사용하기 좋은 은은한 향수"
- "여름 휴가철에 어울리는 시원한 향수"

### 상세 조건
- "20대 초반 여성을 위한 청량감 있는 향수"
- "50대 남성에게 어울리는 고급스러운 향수"
- "향이 오래 지속되는 겨울용 향수"

## 🔧 고급 설정

### 포트 변경
```bash
streamlit run app_improved.py --server.port 8502
```

### 외부 접속 허용
```bash
streamlit run app_improved.py --server.address 0.0.0.0
```

## ❗ 문제 해결

### 연결 오류
1. **Neo4j 서버 확인**
   - Neo4j Desktop에서 데이터베이스가 실행 중인지 확인
   - 연결 정보(.env)가 올바른지 확인

2. **환경변수 확인**
   - `.env` 파일 존재 여부 확인
   - 모든 필수 환경변수 설정 확인

3. **포트 충돌**
   - 8501 포트가 이미 사용 중인 경우 다른 포트 사용

### 모델 로딩 오류
1. **GPU 메모리 부족**
   - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` 설정
   - 다른 GPU 프로세스 종료

2. **HuggingFace 토큰**
   - 유효한 HF_TOKEN 설정 확인
   - 모델 접근 권한 확인

### 성능 최적화
1. **메모리 최적화**
   - `@st.cache_resource` 데코레이터 활용
   - 불필요한 모델 로딩 방지

2. **응답 속도 개선**
   - GPU 사용 확인
   - 배치 크기 조정

## 📊 시스템 아키텍처

```
사용자 입력 → Streamlit UI → GraphRAG 클래스 → Neo4j 검색 → LLM 생성 → 결과 표시
```

### 주요 컴포넌트
- **Frontend**: Streamlit 웹 인터페이스
- **Backend**: GraphRAG 시스템
- **Database**: Neo4j 그래프 데이터베이스
- **Model**: Fine-tuned Llama 3.1 8B

## 📄 파일 구조

```
hyu_llm_2025/
├── script/
│   ├── app_improved.py      # 개선된 Streamlit 앱
│   ├── app.py              # 기본 Streamlit 앱
│   ├── graph_rag.py        # GraphRAG 메인 클래스
│   ├── retrieval.py        # Neo4j 검색 모듈
│   └── test.py             # 연결 테스트
├── data/                   # 데이터 파일들
├── requirements.txt        # 패키지 의존성
├── run_app.sh             # 자동 실행 스크립트
└── README_streamlit.md     # 이 파일
```

## 🎯 향후 개선 계획

- [ ] 향수 리뷰 분석 기능
- [ ] 사용자 선호도 학습
- [ ] 다국어 지원
- [ ] 모바일 앱 개발
- [ ] 향수 비교 기능

## 📞 지원

문제가 발생하면 다음을 확인해주세요:
1. 시스템 요구사항 충족 여부
2. 환경변수 설정 상태
3. Neo4j 데이터베이스 연결 상태
4. 로그 메시지 확인

---

**© 2025 향수 추천 LLM | Powered by Graph RAG & Llama 3.1** 