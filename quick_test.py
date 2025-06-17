#!/usr/bin/env python3
"""
향수 추천 LLM 환경 테스트 스크립트
실행 전 필수 환경을 빠르게 확인합니다.
"""

import os
import sys
from dotenv import load_dotenv

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"🐍 Python 버전: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    print("✅ Python 버전 OK")
    return True

def check_packages():
    """필수 패키지 설치 확인"""
    required_packages = [
        'streamlit',
        'neo4j',
        'transformers',
        'torch',
        'peft',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                from dotenv import load_dotenv
            else:
                __import__(package)
            print(f"✅ {package} 설치됨")
        except ImportError:
            print(f"❌ {package} 미설치")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 다음 패키지를 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 패키지 설치 확인")
    return True

def check_env_file():
    """환경변수 파일 확인"""
    if not os.path.exists('.env'):
        print("❌ .env 파일이 없습니다.")
        print("💡 다음 환경변수를 설정해주세요:")
        print("""
# .env 파일 예시
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TOKEN=your_token
        """)
        return False
    
    print("✅ .env 파일 존재")
    
    # 환경변수 로드 및 확인
    load_dotenv()
    
    required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'MODEL_ID']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var} 미설정")
        else:
            print(f"✅ {var} 설정됨")
    
    if missing_vars:
        print(f"\n🔧 다음 환경변수를 .env 파일에 추가해주세요:")
        for var in missing_vars:
            print(f"{var}=your_value_here")
        return False
    
    return True

def check_neo4j_connection():
    """Neo4j 연결 테스트"""
    try:
        from neo4j import GraphDatabase
        
        load_dotenv()
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection test' as message")
            record = result.single()
            if record:
                print("✅ Neo4j 연결 성공")
                driver.close()
                return True
    except Exception as e:
        print(f"❌ Neo4j 연결 실패: {e}")
        print("💡 Neo4j 서버가 실행 중인지 확인하고 연결 정보를 확인해주세요.")
        return False

def check_gpu():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name} ({gpu_count}개)")
            return True
        else:
            print("⚠️ GPU 사용 불가 (CPU 모드로 실행됩니다)")
            return True
    except:
        print("❌ PyTorch GPU 확인 실패")
        return False

def main():
    """메인 테스트 함수"""
    print("🌸 향수 추천 LLM 환경 테스트 시작")
    print("=" * 50)
    
    all_checks = []
    
    # 각 테스트 실행
    all_checks.append(check_python_version())
    print()
    
    all_checks.append(check_packages())
    print()
    
    all_checks.append(check_env_file())
    print()
    
    all_checks.append(check_neo4j_connection())
    print()
    
    all_checks.append(check_gpu())
    print()
    
    # 전체 결과
    print("=" * 50)
    if all(all_checks):
        print("🎉 모든 테스트 통과! 시스템 준비 완료!")
        print("🚀 다음 명령어로 앱을 실행하세요:")
        print("   ./run_app.sh")
        print("   또는")
        print("   cd script && streamlit run app_improved.py")
    else:
        print("❌ 일부 테스트 실패. 위의 안내를 따라 문제를 해결해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    main() 