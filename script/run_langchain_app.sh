#!/bin/bash

# 🚀 LangChain Graph-RAG 향수 추천 시스템 실행 스크립트

echo "🌸 LangChain Graph-RAG 향수 추천 시스템"
echo "========================================================"

# 현재 디렉토리를 script 폴더로 변경
cd "$(dirname "$0")"

# 가상환경 활성화 확인
if [[ "$CONDA_DEFAULT_ENV" != "yjllm" ]]; then
    echo "⚠️  가상환경을 활성화해주세요: conda activate yjllm"
    exit 1
fi

echo "✅ 가상환경 확인: $CONDA_DEFAULT_ENV"

# .env 파일 존재 확인
if [ ! -f "../.env" ]; then
    echo "❌ .env 파일이 없습니다. 환경변수를 설정해주세요."
    echo "💡 다음 내용으로 .env 파일을 생성하세요:"
    echo ""
    echo "NEO4J_URI=bolt://localhost:7687"
    echo "NEO4J_USER=neo4j"
    echo "NEO4J_PASSWORD=your_password"
    echo "MODEL_ID=path/to/your/llama/model"
    echo "HF_TOKEN=your_huggingface_token"
    exit 1
fi

echo "✅ 환경변수 파일 확인"

# 필요한 패키지 설치
echo "📦 패키지 의존성 확인 중..."
pip install -q -r ../requirements.txt

# 실행 모드 선택
echo ""
echo "📋 실행 모드를 선택하세요:"
echo "1. 🖥️  콘솔 모드 (CLI)"
echo "2. 🌐 웹 앱 모드 (Streamlit)"
echo "3. 🧪 연결 테스트만"
echo ""

read -p "선택 (1-3): " choice

case $choice in
    1)
        echo "🖥️  콘솔 모드로 실행합니다..."
        python langchain_main.py
        ;;
    2)
        echo "🌐 Streamlit 웹 앱을 실행합니다..."
        echo "💡 브라우저에서 http://localhost:8501 로 접속하세요"
        streamlit run langchain_streamlit_app.py
        ;;
    3)
        echo "🧪 연결 테스트를 실행합니다..."
        python langchain_main.py --test
        ;;
    *)
        echo "❓ 올바른 선택이 아닙니다. 1, 2, 3 중에서 선택해주세요."
        exit 1
        ;;
esac

echo ""
echo "🎉 실행 완료!" 