"""
CHA University 학칙/규정 RAG 챗봇 v2
=====================================
- GPT가 검색 전략을 판단 (키워드 추출 + 검색 방식 결정)
- 키워드 전수 검색 + 벡터 유사도 검색 하이브리드
- OpenAI 임베딩 + GPT-4o 답변 생성 + ChromaDB 벡터 저장소
"""

import os
import sys
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ============================================================
# 설정
# ============================================================
CHUNKS_PATH = "chunks.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "cha_regulations"

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ============================================================
# 1. 인덱싱
# ============================================================
def build_index():
    print("인덱싱 시작...")
    if not OPENAI_API_KEY:
        print("[오류] OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  청크 수: {len(chunks)}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            ids=[f"{i+j}" for j, c in enumerate(batch)],
            documents=[c["text"] for c in batch],
            metadatas=[{"document": c["document"]} for c in batch],
        )
        print(f"  [{i+len(batch)}/{len(chunks)}] 임베딩 완료")

    print(f"\n인덱싱 완료! 총 {collection.count()}개 벡터 저장됨")


# ============================================================
# 2. GPT 검색 전략 판단
# ============================================================
def analyze_query(query):
    """GPT가 질문을 분석하여 검색 전략을 결정"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """당신은 대학교 학칙/규정 검색 시스템의 질문 분석기입니다.
사용자의 질문을 분석하여 최적의 검색 전략을 JSON으로 반환하세요.

반드시 아래 JSON 형식만 출력하세요 (다른 텍스트 없이):
{
  "search_type": "keyword" 또는 "vector" 또는 "both",
  "keywords": ["검색할", "키워드", "목록"],
  "vector_query": "벡터 검색에 사용할 자연어 질문",
  "reason": "판단 이유 한 줄"
}

판단 기준:
- "~가 들어간/포함된/언급된 규정 찾아줘" → keyword (해당 단어를 정확히 포함하는 조항 전수 검색)
- "~에 대한 규정이 뭐야?", "~하려면 어떻게 해?" → vector (의미 기반 검색)
- "~관련 규정 모두 찾아줘" → both (키워드 + 벡터 병행)

keywords에는 실제 규정 본문에서 검색할 핵심 단어만 넣으세요.
"규정", "조항", "학칙" 같은 메타 단어는 절대 넣지 마세요.
사용자가 찾고자 하는 실질적 내용어만 추출하세요.

예시:
- "AI라는 용어가 들어간 규정 찾아줘" → keywords: ["AI"]
- "인공지능, 데이터, 디지털 관련 조항 모두 찾아줘" → keywords: ["인공지능", "데이터", "디지털", "AI"]
- "졸업하려면 어떤 요건을 충족해야 해?" → search_type: "vector", vector_query: "졸업 요건 학점 이수"
- "장학금 관련 규정에서 성적 기준 찾아줘" → search_type: "both", keywords: ["장학금", "성적"], vector_query: "장학금 성적 기준"
"""},
            {"role": "user", "content": query}
        ],
        max_tokens=300,
        temperature=0,
    )

    try:
        result_text = response.choices[0].message.content.strip()
        # JSON 블록 추출
        if '```' in result_text:
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        return json.loads(result_text)
    except:
        # 파싱 실패 시 기본값
        return {
            "search_type": "vector",
            "keywords": [],
            "vector_query": query,
            "reason": "질문 분석 실패, 벡터 검색으로 대체"
        }


# ============================================================
# 3. 검색 엔진
# ============================================================
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )


def load_all_chunks():
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def keyword_search(keywords):
    """키워드 기반 전수 검색"""
    chunks = load_all_chunks()
    results = []

    for chunk in chunks:
        text = chunk['text']
        text_lower = text.lower()
        matched = []
        for kw in keywords:
            count = text_lower.count(kw.lower())
            if count > 0:
                matched.append((kw, count))

        if matched:
            score = sum(c for _, c in matched)
            results.append({
                'text': text,
                'document': chunk['document'],
                'matched_keywords': [k for k, _ in matched],
                'score': score
            })

    # 점수순 정렬
    results.sort(key=lambda x: -x['score'])
    return results


def vector_search(query, n_results=15):
    """벡터 유사도 검색"""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results


def execute_search(strategy, n_results=15):
    """검색 전략에 따라 검색 실행"""
    search_type = strategy.get('search_type', 'vector')
    keywords = strategy.get('keywords', [])
    vector_query = strategy.get('vector_query', '')

    all_docs = []
    all_metas = []
    search_info = {}

    # 키워드 검색
    if search_type in ('keyword', 'both') and keywords:
        kw_results = keyword_search(keywords)
        search_info['keyword_total'] = len(kw_results)

        # 문서별 그룹핑하여 통계
        doc_set = set(r['document'] for r in kw_results)
        search_info['keyword_docs'] = len(doc_set)
        search_info['keywords'] = keywords

        # 상위 결과 추가 (점수순으로 최대 30개)
        seen_texts = set()
        for r in kw_results[:30]:
            text_key = r['text'][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_docs.append(r['text'])
                all_metas.append({
                    'document': r['document'],
                    'matched_keywords': r['matched_keywords']
                })

    # 벡터 검색
    if search_type in ('vector', 'both'):
        query = vector_query if vector_query else ' '.join(keywords)
        if query:
            vec_results = vector_search(query, n_results=n_results)
            seen_texts = set(d[:100] for d in all_docs)

            if vec_results['documents'] and vec_results['documents'][0]:
                for doc, meta in zip(vec_results['documents'][0], vec_results['metadatas'][0]):
                    if doc[:100] not in seen_texts:
                        seen_texts.add(doc[:100])
                        all_docs.append(doc)
                        all_metas.append(meta)

    search_info['search_type'] = search_type
    search_info['total_context'] = len(all_docs)

    return {
        'documents': [all_docs],
        'metadatas': [all_metas],
        'search_info': search_info,
    }


# ============================================================
# 4. 답변 생성
# ============================================================
def generate_answer(query, search_results, strategy):
    """GPT-4o로 답변 생성"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    docs = search_results['documents'][0] if search_results['documents'] else []
    metas = search_results['metadatas'][0] if search_results['metadatas'] else []
    search_info = search_results.get('search_info', {})

    # 컨텍스트 구성 (최대 25개)
    context_parts = []
    for doc_text, meta in zip(docs[:25], metas[:25]):
        doc_name = meta.get('document', '알수없음')
        keywords = meta.get('matched_keywords', [])
        kw_info = f" [매칭: {', '.join(keywords)}]" if keywords else ""
        context_parts.append(f"[규정: {doc_name}]{kw_info}\n{doc_text}")

    context = "\n\n---\n\n".join(context_parts)

    # 검색 통계
    stats_parts = []
    if 'keywords' in search_info:
        stats_parts.append(f"키워드 {search_info['keywords']}로 검색")
    if 'keyword_total' in search_info:
        stats_parts.append(f"{search_info.get('keyword_docs', 0)}개 규정에서 {search_info['keyword_total']}개 조항 발견")
    stats = f"\n[검색 정보: {', '.join(stats_parts)}]" if stats_parts else ""

    system_prompt = """당신은 차의과학대학교(CHA University)의 학칙 및 규정 전문 어시스턴트입니다.

역할:
- 사용자의 질문에 대해 제공된 규정 문서를 기반으로 정확하게 답변합니다.
- 답변 시 반드시 근거가 되는 규정명과 조항을 명시합니다.
- 규정에 없는 내용은 "해당 규정에서 관련 내용을 찾지 못했습니다"라고 솔직하게 답합니다.
- 여러 규정에 걸쳐 관련 내용이 있으면 종합적으로 정리합니다.
- 특정 키워드가 포함된 조항을 찾는 질문의 경우, 해당 키워드가 실제로 등장하는 모든 조항을 빠짐없이 나열합니다.

답변 형식:
- 핵심 답변을 먼저 제시
- 근거 규정 및 조항 번호 명시
- 전수 조사의 경우 규정별로 정리하여 표시
- 필요시 관련 규정 간 연관성 설명"""

    user_message = f"""다음은 질문과 관련된 차의과학대학교 규정 내용입니다:
{stats}

{context}

---

질문: {query}

위 규정 내용을 바탕으로 정확하고 빠짐없이 답변해주세요."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=4000,
    )

    return response.choices[0].message.content


# ============================================================
# 5. Streamlit UI
# ============================================================
def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="CHA 학칙/규정 챗봇",
        page_icon="🎓",
        layout="wide"
    )

    st.title("CHA University 학칙/규정 챗봇")
    st.caption("차의과학대학교 학칙 및 규정을 기반으로 답변합니다 (136개 규정, 2,059개 조항)")

    with st.sidebar:
        st.header("설정")
        n_results = st.slider("벡터 검색 결과 수", 5, 30, 15)
        show_sources = st.checkbox("참조 규정 표시", value=True)
        show_strategy = st.checkbox("검색 전략 표시", value=True)

        st.divider()
        st.header("검색 방식")
        st.caption("""
        **GPT가 자동 판단:**
        - 키워드 전수 검색
        - 벡터 유사도 검색  
        - 하이브리드 (둘 다)
        """)

        st.divider()
        try:
            collection = get_collection()
            st.metric("총 벡터 수", collection.count())
        except:
            st.warning("인덱스가 없습니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("학칙/규정에 대해 질문하세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("질문을 분석하고 검색 중..."):
                try:
                    # 1단계: GPT가 검색 전략 판단
                    strategy = analyze_query(prompt)

                    if show_strategy:
                        search_type = strategy.get('search_type', 'vector')
                        keywords = strategy.get('keywords', [])
                        reason = strategy.get('reason', '')
                        
                        type_labels = {
                            'keyword': '🔤 키워드 전수 검색',
                            'vector': '🧠 벡터 유사도 검색',
                            'both': '🔍 하이브리드 검색'
                        }
                        label = type_labels.get(search_type, search_type)
                        
                        info_text = f"{label}"
                        if keywords:
                            info_text += f" | 키워드: {keywords}"
                        if reason:
                            info_text += f"\n{reason}"
                        st.info(info_text)

                    # 2단계: 검색 실행
                    results = execute_search(strategy, n_results=n_results)
                    search_info = results.get('search_info', {})

                    if search_info.get('keyword_total'):
                        st.caption(f"📊 {search_info.get('keyword_docs', 0)}개 규정에서 "
                                   f"{search_info['keyword_total']}개 조항 발견 → "
                                   f"상위 {search_info.get('total_context', 0)}개로 답변 생성")

                    # 3단계: 답변 생성
                    answer = generate_answer(prompt, results, strategy)
                    st.markdown(answer)

                    # 참조 규정
                    if show_sources and results['metadatas'][0]:
                        doc_count = len(results['documents'][0])
                        with st.expander(f"참조 규정 보기 ({doc_count}개 조항)"):
                            seen_docs = set()
                            for meta, text in zip(results['metadatas'][0], results['documents'][0]):
                                doc_name = meta.get('document', '알수없음')
                                if doc_name not in seen_docs:
                                    seen_docs.add(doc_name)
                                    st.markdown(f"**📄 {doc_name}**")
                                keywords = meta.get('matched_keywords', [])
                                if keywords:
                                    st.caption(f"매칭 키워드: {', '.join(keywords)}")
                                display_text = text if isinstance(text, str) else str(text)
                                # 키워드 하이라이트
                                if keywords:
                                    highlighted = display_text
                                    for kw in keywords:
                                        highlighted = highlighted.replace(kw, f"**🔴 {kw}**")
                                    st.markdown(highlighted)
                                else:
                                    st.text(display_text)
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    if "--index" in sys.argv:
        build_index()
    else:
        run_streamlit()
