#!/bin/bash

# 향수 추천 LLM 실행 스크립트

echo "🌸 향수 추천 LLM 시작 중..."

# Conda 환경 활성화
echo "🔧 yjllm 가상환경을 활성화합니다..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yjllm

# 환경변수 파일 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다."
    echo "💡 환경변수를 설정해주세요:"
    echo "   NEO4J_URI=bolt://localhost:7687"
    echo "   NEO4J_USER=neo4j"
    echo "   NEO4J_PASSWORD=your_password"
    echo "   MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "   HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Python 환경 확인
if ! command -v python &> /dev/null; then
    echo "❌ Python이 설치되어 있지 않습니다."
    exit 1
fi

# 패키지 설치 확인 및 설치
echo "📦 필요한 패키지들을 확인하고 설치합니다..."
pip install -r requirements.txt

# Neo4j 연결 테스트
echo "🔗 Neo4j 연결을 테스트합니다..."
cd script
python test.py

# Streamlit 앱 실행
echo "🚀 Streamlit 앱을 실행합니다..."
echo "   접속 주소: http://localhost:8501"
streamlit run app_improved.py --server.port 8501 --server.address 0.0.0.0 