import streamlit as st
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from langchain_graph_rag_fixed import PerfumeRecommendationSystem

# 페이지 설정
st.set_page_config(
    page_title="🌸 향수 추천 시스템 (Graph-RAG)",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바에 DB 선택 추가
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    # DB 버전 선택
    db_version = st.selectbox(
        "🗄️ 데이터베이스 버전",
        options=["기존 DB", "새 DB (v2)"],
        help="비교할 데이터베이스를 선택하세요"
    )
    
    # DB 설정
    if db_version == "기존 DB":
        db_config = {
            "uri": "bolt://166.104.90.18:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
        }
        st.info("📋 기존 데이터베이스 사용 중")
    else:
        db_config = {
            "uri": "bolt://166.104.90.18:7687",  # 같은 서버
            "username": "neo4j",
            "password": "password",
            "database": "perfume_v2"  # 다른 데이터베이스
        }
        st.success("🆕 새 데이터베이스 (v2) 사용 중 - DB: perfume_v2")
    
    st.divider()
    
    # LLM 설정
    st.subheader("🤖 LLM 설정")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 1000, 300, 50)
    top_k = st.slider("검색 결과 수", 3, 10, 5, 1)

# 시스템 초기화 (DB 설정 적용)
@st.cache_resource
def init_system(db_uri, db_username, db_password, db_database):
    return PerfumeRecommendationSystem(
        neo4j_uri=db_uri,
        neo4j_username=db_username,
        neo4j_password=db_password,
        database=db_database
    )

# 선택된 DB로 시스템 초기화
system = init_system(
    db_config["uri"],
    db_config["username"], 
    db_config["password"],
    db_config["database"]
)

# 메인 헤더
st.title("🌸 향수 추천 시스템")
st.markdown(f"**Graph-RAG 기반 개인 맞춤형 향수 추천** ({db_version})")

# 샘플 질문들
st.subheader("💡 샘플 질문 (클릭해보세요!)")
sample_questions = [
    "남자 우디 계열 샤넬 향수",
    "여성용 플로럴 향수 추천해줘",
    "톰포드 브랜드 남성 향수",
    "겨울에 어울리는 향수",
    "오피스에서 쓸 수 있는 은은한 향수"
]

cols = st.columns(len(sample_questions))
for i, question in enumerate(sample_questions):
    with cols[i]:
        if st.button(f"📝 {question}", key=f"sample_{i}"):
            st.session_state.user_question = question

# 사용자 입력
user_input = st.text_input(
    "💬 향수에 대해 무엇이든 물어보세요:",
    value=st.session_state.get('user_question', ''),
    placeholder="예: 남자 우디 계열 샤넬 향수",
    key="current_input"
)

# 입력 초기화 버튼
if st.button("🗑️ 입력 초기화"):
    st.session_state.user_question = ""
    st.rerun()

# 질문 처리
if user_input:
    with st.spinner("🔍 향수를 찾고 있습니다..."):
        try:
            # 추천 시스템 실행
            result = system.get_recommendation(
                user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k
            )
            
            # 결과 표시
            st.success("✨ 추천 완료!")
            
            # 탭으로 결과 구분
            tab1, tab2, tab3 = st.tabs(["🎯 추천 결과", "📊 검색 과정", "🔧 디버그 정보"])
            
            with tab1:
                st.markdown("### 🌟 추천 향수")
                st.markdown(result["response"])
                
                # 추가 정보가 있다면 표시
                if "metadata" in result:
                    st.markdown("### 📋 추가 정보")
                    st.json(result["metadata"])
            
            with tab2:
                st.markdown("### 🔍 검색된 향수들")
                if "retrieved_perfumes" in result:
                    for i, perfume in enumerate(result["retrieved_perfumes"], 1):
                        with st.expander(f"#{i} {perfume.get('name', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**브랜드:** {perfume.get('brand', 'N/A')}")
                                st.write(f"**타겟:** {perfume.get('target', 'N/A')}")
                                if db_version == "새 DB (v2)" and perfume.get('short_description'):
                                    st.write(f"**설명:** {perfume.get('short_description')}")
                            
                            with col2:
                                st.write(f"**평점:** {perfume.get('rating', 'N/A')}")
                                st.write(f"**리뷰 수:** {perfume.get('review_count', 'N/A')}")
                                st.write(f"**유사도:** {perfume.get('similarity', 'N/A')}")
            
            with tab3:
                st.markdown("### 🔧 시스템 정보")
                st.write(f"**사용 DB:** {db_version}")
                st.write(f"**Temperature:** {temperature}")
                st.write(f"**Max Tokens:** {max_tokens}")
                st.write(f"**검색 결과 수:** {top_k}")
                
                if "debug_info" in result:
                    st.json(result["debug_info"])
                    
        except Exception as e:
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
            st.error("시스템 관리자에게 문의해주세요.")

# 대화 기록
if "chat_history" in st.session_state and st.session_state.chat_history:
    with st.expander("📜 대화 기록", expanded=False):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.divider()

# 푸터 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 연결 정보**")
    st.text(f"Neo4j: {db_config['uri']}")
    st.text(f"DB: {db_config['database']}")

with col2:
    st.markdown("**📊 성능 정보**")
    st.text("BGE-M3 임베딩")
    st.text("Graph-RAG 검색")

with col3:
    st.markdown("**🆕 새 기능**")
    if db_version == "새 DB (v2)":
        st.text("✅ 향상된 임베딩")
        st.text("✅ 설명 정보 포함")
    else:
        st.text("📋 기본 기능")
        st.text("📋 표준 임베딩") 