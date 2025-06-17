import streamlit as st
import sys
import os
from datetime import datetime
import warnings

# PyTorch 관련 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Streamlit 설정을 먼저 하고 PyTorch 관련 모듈 import
try:
    from langchain_graph_rag_fixed import PerfumeRecommendationSystem
except Exception as e:
    st.error(f"모듈 로딩 오류: {str(e)}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="🌸 LangChain 향수 추천 시스템",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    # DB 설정 (단일 데이터베이스 사용)
    db_config = {
        "uri": "bolt://166.104.90.18:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j"
    }
    st.info("📋 향수 데이터베이스 연결 중 (Perfume, Brand, Target, Accord)")
    
    st.divider()
    
    # LLM 설정
    st.subheader("🤖 LLM 파라미터")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, 
                           help="응답의 창의성을 조절합니다. 낮을수록 일관된 답변")
    max_tokens = st.slider("Max Tokens", 100, 1000, 300, 50,
                          help="생성할 최대 토큰 수")
    top_k = st.slider("검색 결과 수", 3, 10, 5, 1,
                     help="유사한 향수를 몇 개까지 검색할지 설정")

# 시스템 초기화
@st.cache_resource
def init_system(db_uri, db_username, db_password, db_database):
    """향수 추천 시스템을 초기화합니다."""
    try:
        system = PerfumeRecommendationSystem(
            neo4j_uri=db_uri,
            neo4j_user=db_username,
            neo4j_password=db_password,
            database=db_database
        )
        return system
    except Exception as e:
        st.error(f"시스템 초기화 실패: {str(e)}")
        return None

# 시스템 초기화
system = init_system(
    db_config["uri"],
    db_config["username"], 
    db_config["password"],
    db_config["database"]
)

# 메인 헤더
st.title("🌸 LangChain 향수 추천 시스템")
st.markdown("**Graph-RAG 기반 개인 맞춤형 향수 추천**")

# 시스템 상태 확인
if system is None:
    st.error("❌ 시스템 초기화에 실패했습니다. 설정을 확인해주세요.")
    st.stop()

# 사용자 입력 (맨 위로 이동)
st.subheader("💬 향수 질문하기")

# Form을 사용하여 Enter 키로도 검색 가능하도록 설정
with st.form(key="search_form", clear_on_submit=False):
    # 입력창과 버튼을 한 줄에 배치
    input_col, button_col1, button_col2 = st.columns([6, 1.2, 1.2])
    
    with input_col:
        user_input = st.text_input(
            "질문을 입력하세요:",
            value=st.session_state.get('user_question', ''),
            placeholder="예: 남자 우디 계열 샤넬 향수 추천해줘",
            label_visibility="collapsed"
        )
    
    with button_col1:
        search_button = st.form_submit_button("🔍 검색", type="primary", use_container_width=True)
    
    with button_col2:
        clear_button = st.form_submit_button("🗑️ 초기화", use_container_width=True)

# 초기화 버튼 처리
if clear_button:
    if 'user_question' in st.session_state:
        del st.session_state.user_question
    if 'auto_search' in st.session_state:
        del st.session_state.auto_search
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
    st.rerun()

st.divider()

# 질문 처리 (검색 버튼 클릭 또는 샘플 질문 선택 시)
if user_input and (search_button or st.session_state.get('auto_search', False)):
    # LLM 파라미터 실시간 업데이트
    system.update_generation_settings(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.7  # top_p는 고정값 사용
    )
    
    # 진행 바와 상태 메시지를 위한 컨테이너
    progress_container = st.container()
    
    with progress_container:
        # 진행 바 초기화
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 단계 1: 키워드 추출
            status_text.text("🔍 1/5 단계: 사용자 질문 분석 중...")
            progress_bar.progress(20)
            
            # 단계 2: 임베딩 검색
            status_text.text("🧠 2/5 단계: 임베딩 벡터 검색 중...")
            progress_bar.progress(40)
            
            # 단계 3: 그래프 검색
            status_text.text("🕸️ 3/5 단계: 그래프 데이터베이스 검색 중...")
            progress_bar.progress(60)
            
            # 단계 4: LLM 생성
            status_text.text("🤖 4/5 단계: AI 답변 생성 중...")
            progress_bar.progress(80)
            
            # 실제 추천 시스템 실행
            result = system.get_recommendation(
                user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k
            )
            
            # 단계 5: 완료
            status_text.text("✨ 5/5 단계: 추천 완료!")
            progress_bar.progress(100)
            
            # 잠시 완료 메시지 표시 후 진행 바 제거
            import time
            time.sleep(1)
            progress_container.empty()
            
            # 성공 메시지
            st.success("✨ 향수 추천이 완료되었습니다!")
            
            # 답변 섹션
            st.subheader("🌟 추천 답변")
            
            # 메인 답변을 카드 형태로 표시
            with st.container():
                st.markdown("""
                <div style="
                    background-color: #f0f8ff;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #4CAF50;
                    margin: 10px 0;
                ">
                """, unsafe_allow_html=True)
                
                st.markdown(result["response"])
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 상세 정보를 탭으로 구분
            tab1, tab2, tab3 = st.tabs(["📊 검색된 향수", "🔍 검색 과정", "🔧 시스템 정보"])
            
            with tab1:
                if "retrieved_perfumes" in result and result["retrieved_perfumes"]:
                    st.markdown(f"**총 {len(result['retrieved_perfumes'])}개의 향수를 찾았습니다.**")
                    
                    for i, perfume in enumerate(result["retrieved_perfumes"], 1):
                        with st.expander(f"#{i} {perfume.get('name', 'Unknown')} ({perfume.get('brand', 'N/A')})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**🏷️ 브랜드:** {perfume.get('brand', 'N/A')}")
                                st.write(f"**👥 타겟:** {perfume.get('target', 'N/A')}")
                                if perfume.get('short_description'):
                                    st.write(f"**📝 설명:** {perfume.get('short_description')}")
                                if perfume.get('notes'):
                                    st.write(f"**🎵 노트:** {perfume.get('notes')}")
                            
                            with col2:
                                st.write(f"**⭐ 평점:** {perfume.get('rating', 'N/A')}")
                                st.write(f"**💬 리뷰 수:** {perfume.get('review_count', 'N/A')}")
                                st.write(f"**🎯 유사도:** {perfume.get('similarity', 'N/A')}")
                                if perfume.get('price'):
                                    st.write(f"**💰 가격:** {perfume.get('price')}")
                else:
                    st.warning("검색된 향수가 없습니다.")
            
            with tab2:
                # 쿼리 정보
                st.markdown("#### 📝 사용자 쿼리")
                st.code(user_input)
                
                # 검색 통계
                if "search_stats" in result:
                    st.markdown("#### 📊 검색 통계")
                    stats = result["search_stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("검색된 향수 수", stats.get("total_found", 0))
                    with col2:
                        st.metric("응답 시간", f"{stats.get('response_time', 0):.2f}초")
                    with col3:
                        st.metric("사용된 토큰", stats.get("tokens_used", 0))
                
                # 디버그 정보
                if "debug_info" in result:
                    st.markdown("#### 🔧 디버그 정보")
                    with st.expander("상세 디버그 정보"):
                        st.json(result["debug_info"])
            
            with tab3:
                # 현재 설정
                settings_col1, settings_col2 = st.columns(2)
                
                with settings_col1:
                    st.markdown("#### 🗄️ 데이터베이스")
                    st.write(f"**URI:** {db_config['uri']}")
                    st.write(f"**Database:** {db_config['database']}")
                
                with settings_col2:
                    st.markdown("#### 🤖 LLM 설정")
                    st.write(f"**Temperature:** {temperature}")
                    st.write(f"**Max Tokens:** {max_tokens}")
                    st.write(f"**Top K:** {top_k}")
                
                # 시스템 성능
                st.markdown("#### 📊 시스템 성능")
                st.success("✅ BGE-M3 임베딩 모델")
                st.success("✅ Graph-RAG 검색")
                st.success("✅ LangChain 프레임워크")
            
            # 대화 기록 저장
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # 새 대화 추가 (중복 방지)
            if not st.session_state.chat_history or st.session_state.chat_history[-1]["question"] != user_input:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append({
                    "timestamp": current_time,
                    "question": user_input,
                    "answer": result.get("response", "응답 없음"),
                    "perfume_count": len(result.get("retrieved_perfumes", []))
                })
                    
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
            st.error("네트워크 연결이나 데이터베이스 상태를 확인해주세요.")
            
            # 에러 상세 정보
            with st.expander("🔍 에러 상세 정보"):
                st.code(str(e))
    
    # 검색 완료 후 세션 상태 정리
    if 'user_question' in st.session_state:
        del st.session_state.user_question
    if 'auto_search' in st.session_state:
        del st.session_state.auto_search

# 대화 기록 표시 (답변 바로 아래)
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.divider()
    st.subheader("📜 대화 기록")
    
    # 대화 기록을 카드 형태로 표시
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # 최근 3개만 표시
        chat_num = len(st.session_state.chat_history) - i
        
        with st.expander(f"💬 대화 {chat_num} - {chat['timestamp']} (향수 {chat['perfume_count']}개 검색)", expanded=(i==0)):
            # 질문
            st.markdown("**🙋‍♀️ 질문:**")
            st.markdown(f"*{chat['question']}*")
            
            # 답변
            st.markdown("**🤖 답변:**")
            st.markdown(chat['answer'])
    
    # 전체 대화 기록 보기
    if len(st.session_state.chat_history) > 3:
        with st.expander(f"📋 전체 대화 기록 보기 (총 {len(st.session_state.chat_history)}개)"):
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"**대화 {i+1}** - {chat['timestamp']}")
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer'][:100]}{'...' if len(chat['answer']) > 100 else ''}")
                st.markdown("---")
    
    # 대화 기록 관리 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("💾 대화 기록 내보내기", use_container_width=True):
            # JSON 형태로 대화 기록 다운로드
            import json
            chat_data = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 파일 다운로드",
                data=chat_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

st.divider()

# 샘플 질문들 (대화 기록 아래로 이동)
st.subheader("💡 샘플 질문")
st.markdown("*아래 질문들을 클릭해서 바로 검색해보세요!*")

sample_questions = [
    "남자 우디 계열 샤넬 향수 추천해줘",
    "여성용 플로럴 향수 중에 인기 있는 것은?",
    "톰포드 브랜드 남성 향수 어떤 게 좋아?",
    "겨울에 어울리는 따뜻한 향수",
    "오피스에서 쓸 수 있는 은은한 향수",
    "20대 여성에게 어울리는 향수",
    "데이트할 때 쓰기 좋은 향수"
]

# 샘플 질문을 여러 줄로 배치
cols_per_row = 2  # 한 줄에 2개씩 배치하여 버튼을 더 크게
for i in range(0, len(sample_questions), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < len(sample_questions):
            question = sample_questions[i + j]
            with col:
                if st.button(f"📝 {question}", key=f"sample_{i+j}", use_container_width=True):
                    st.session_state.user_question = question
                    st.session_state.auto_search = True
                    st.rerun()

# 푸터 정보
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 연결 정보**")
    st.text(f"Neo4j: {db_config['uri']}")
    st.text(f"DB: {db_config['database']}")
    st.text(f"Status: {'🟢 연결됨' if system else '🔴 연결 실패'}")

with footer_col2:
    st.markdown("**📊 기술 스택**")
    st.text("🔍 BGE-M3 임베딩")
    st.text("🕸️ Graph-RAG 검색")
    st.text("🤖 LangChain 프레임워크")

with footer_col3:
    st.markdown("**🆕 기능**")
    st.text("✅ 향수 추천")
    st.text("✅ 그래프 검색")
    st.text("✅ 자연어 처리")

# 사이드바 하단에 도움말
with st.sidebar:
    st.divider()
    st.markdown("### 💡 사용 팁")
    st.markdown("""
    - **구체적인 질문**을 해보세요
    - **브랜드명, 성별, 계열**을 포함하면 더 정확합니다
    - **샘플 질문**을 클릭해서 시작해보세요
    - **LLM 파라미터**를 조정해서 다양한 결과를 확인해보세요
    """)
    
    if st.button("🔄 시스템 재시작"):
        st.cache_resource.clear()
        st.rerun() 