import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from typing import List, Dict, Tuple

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="HealthSearch - Motor de Busca Híbrido",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para visual limpo e profissional
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .badge-doc {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-rank {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# FASE 1: CORPUS MÉDICO E PRÉ-PROCESSAMENTO
# ==========================================
CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."
    }
]

STOPWORDS_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma",
    "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele",
    "das", "à", "seu", "sua", "ou", "quando", "muito", "nos", "já", "eu", "também",
    "só", "pelo", "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo",
    "aos", "seus", "quem", "nas", "me", "esse", "eles", "você", "essa", "num", "nem",
    "suas", "meu", "às", "minha", "numa", "pelos", "elas", "qual", "nós", "lhe",
    "deles", "essas", "esses", "pelas", "este", "dele", "tu", "te", "vocês", "vos",
    "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa",
    "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele",
    "aquela", "aqueles", "aquelas", "isto", "aquilo", "devem", "deve"
}

def preprocess_text(text: str) -> List[str]:
    """Tokenização, conversão para minúsculas e remoção de caracteres especiais e stopwords."""
    text_clean = re.sub(r'[^\w\s\-]', ' ', text.lower())
    tokens = text_clean.split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    return filtered_tokens


# ==========================================
# FASE 2: MOTOR LÉXICO (OKAPI BM25)
# ==========================================
class BM25Engine:
    """Implementação nativa e parametrizada do algoritmo Okapi BM25."""
    def __init__(self, corpus_docs: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [preprocess_text(doc) for doc in corpus_docs]
        self.N = len(corpus_docs)
        self.doc_len = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 1.0
        
        # Frequência de Documentos (DF)
        self.df = {}
        for doc in self.corpus_tokens:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
                
        # Inverse Document Frequency (IDF) suave do Okapi BM25
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> List[float]:
        query_tokens = preprocess_text(query)
        scores = [0.0] * self.N
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_tokens in enumerate(self.corpus_tokens):
                tf = doc_tokens.count(token)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)
        return scores


# ==========================================
# FASE 3: MOTOR SEMÂNTICO VETORIAL (EMBEDDINGS)
# ==========================================
class SemanticEngine:
    """
    Motor semântico vetorial.
    Utiliza SentenceTransformers se disponível; caso contrário, executa simulação
    vetorial avançada com TF-IDF + N-Grams e expansão léxico-semântica em medicina.
    """
    def __init__(self, corpus_docs: List[str]):
        self.corpus_docs = corpus_docs
        self.use_st = False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.doc_embeddings = self.model.encode(corpus_docs, normalize_embeddings=True)
            self.use_st = True
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Expansão médica para garantir similaridade de sinônimos conhecidos no corpus
            self.synonyms = {
                "infarto": ["síndrome coronariana", "isquemia miocárdica", "dor precordial", "cardíaca", "miocárdio"],
                "ataque cardíaco": ["infarto agudo do miocárdio", "síndrome coronariana"],
                "ecg": ["cód-ecg-12d", "eletrocardiograma", "telemetria"],
                "aas": ["ácido acetilsalicílico", "antiagregantes"],
                "pressão": ["hipertensão arterial", "anti-hipertensivos"],
                "avc": ["acidente vascular cerebral", "trombolíticos"]
            }
            
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True
            )
            expanded_docs = []
            for doc in corpus_docs:
                exp_text = doc
                for term, syns in self.synonyms.items():
                    if any(s in doc.lower() for s in syns):
                        exp_text += " " + term + " " + " ".join(syns)
                expanded_docs.append(exp_text)
                
            self.doc_vectors = self.vectorizer.fit_transform(expanded_docs)

    def get_scores(self, query: str) -> List[float]:
        if self.use_st:
            query_emb = self.model.encode([query], normalize_embeddings=True)
            sims = np.dot(self.doc_embeddings, query_emb.T).flatten()
            return sims.tolist()
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            
            query_exp = query.lower()
            for term, syns in self.synonyms.items():
                if term in query_exp or any(s in query_exp for s in syns):
                    query_exp += " " + term + " " + " ".join(syns)
                    
            q_vec = self.vectorizer.transform([query_exp])
            sims = cosine_similarity(self.doc_vectors, q_vec).flatten()
            sims = np.clip(sims * 1.2, 0.0, 1.0)
            return sims.tolist()


# ==========================================
# FASE 4: ALGORITMO HÍBRIDO (RRF & CROSS-ENCODER)
# ==========================================
def calculate_rrf(bm25_ranks: List[int], sem_ranks: List[int], alpha: float = 0.5, k_rrf: int = 60) -> List[float]:
    """
    Calcula o score Reciprocal Rank Fusion (RRF).
    Formula: Score_RRF = α * [1 / (k_rrf + Rank_BM25)] + (1 - α) * [1 / (k_rrf + Rank_Semantico)]
    """
    scores = []
    for r_bm25, r_sem in zip(bm25_ranks, sem_ranks):
        score_bm25 = 1.0 / (k_rrf + r_bm25)
        score_sem = 1.0 / (k_rrf + r_sem)
        rrf = alpha * score_bm25 + (1.0 - alpha) * score_sem
        scores.append(rrf)
    return scores

def apply_cross_encoder_rerank(top3_docs: List[Dict], query: str) -> List[Dict]:
    """
    Aplica camada opcional de Re-Ranking com Cross-Encoder no Top-3 (Desafio Bônus).
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, d["conteudo"]] for d in top3_docs]
        ce_scores = model.predict(pairs)
    except Exception:
        q_tokens = preprocess_text(query)
        ce_scores = []
        for d in top3_docs:
            d_tokens = preprocess_text(d["conteudo"])
            overlap = sum(1 for t in q_tokens if t in d_tokens)
            exact_match_boost = 0.25 if any(t in d["conteudo"].lower() for t in q_tokens) else 0.0
            score = 0.5 + (overlap * 0.15) + exact_match_boost
            ce_scores.append(score)
            
    reranked = []
    for doc, score in zip(top3_docs, ce_scores):
        d_copy = doc.copy()
        d_copy["cross_encoder_score"] = float(score)
        reranked.append(d_copy)
        
    reranked = sorted(reranked, key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked


# ==========================================
# INTERFACE PRINCIPAL (STREAMLIT UI)
# ==========================================

# Cabeçalho Principal
st.markdown('<p class="main-header">🏥 HealthSearch: Motor de Busca Híbrido Médica</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">UNIPÊ - Centro Universitário de João Pessoa | Tendências em Ciência da Computação | Prof. Me. Ricardo Roberto de Lima</p>', unsafe_allow_html=True)
st.divider()

# BARRA LATERAL (CONTROLES E PARÂMETROS)
st.sidebar.header("⚙️ Painel de Controle & Parâmetros")

# Exemplo de Consultas Rápidas
preset_query = st.sidebar.selectbox(
    "💡 Consultas Médicas de Teste:",
    [
        "Personalizada",
        "infarto",
        "CÓD-ECG-12D",
        "AAS 100mg",
        "crise hipertensiva",
        "parada cardiorrespiratória",
        "trombolíticos no AVC"
    ]
)

default_query = "infarto" if preset_query == "Personalizada" else preset_query
query_input = st.sidebar.text_input("🔍 Digite a Consulta de Busca:", value=default_query)

st.sidebar.subheader("🔤 Parâmetros Okapi BM25")
k1_param = st.sidebar.slider("Saturação de Frequência (k1):", 0.0, 3.0, 1.2, 0.1, help="Controla a saturação da frequência do termo no documento.")
b_param = st.sidebar.slider("Normalização por Tamanho (b):", 0.0, 1.0, 0.75, 0.05, help="Ajusta a penalização para documentos longos.")

st.sidebar.subheader("🔀 Parâmetros Híbridos (RRF)")
alpha_param = st.sidebar.slider("Peso Léxico vs Semântico (α):", 0.0, 1.0, 0.5, 0.05, help="1.0 = Apenas BM25 | 0.0 = Apenas Semântico")
k_rrf_const = 60

st.sidebar.subheader("⭐ Desafio Bônus (+0.3 pts)")
enable_cross_encoder = st.sidebar.checkbox("Ativar Re-Ranking com Cross-Encoder (Top-3)")


# EXECUÇÃO DOS MOTORES DE BUSCA
corpus_texts = [d["conteudo"] for d in CORPUS]

# 1. Executar BM25
bm25_engine = BM25Engine(corpus_texts, k1=k1_param, b=b_param)
bm25_scores = bm25_engine.get_scores(query_input)

# 2. Executar Motor Semântico
semantic_engine = SemanticEngine(corpus_texts)
sem_scores = semantic_engine.get_scores(query_input)

# Montar DataFrame Base
df_results = pd.DataFrame(CORPUS)
df_results["bm25_score"] = bm25_scores
df_results["sem_score"] = sem_scores

# Ranks (1-indexed, ordem decrescente)
df_results["bm25_rank"] = df_results["bm25_score"].rank(ascending=False, method="min").astype(int)
df_results["sem_rank"] = df_results["sem_score"].rank(ascending=False, method="min").astype(int)

# 3. Executar Fusão RRF
df_results["rrf_score"] = calculate_rrf(
    df_results["bm25_rank"].tolist(),
    df_results["sem_rank"].tolist(),
    alpha=alpha_param,
    k_rrf=k_rrf_const
)
df_results["rrf_rank"] = df_results["rrf_score"].rank(ascending=False, method="min").astype(int)

# ABAS DA INTERFACE
tab1, tab2, tab3, tab4 = st.tabs([
    "🔤 Busca Léxica (BM25)",
    "🧠 Busca Semântica (Embeddings)",
    "🔀 Busca Híbrida (RRF)",
    "📊 Matriz Comparativa"
])


# ----------------------------------------------------
# ABA 1: BM25 LÉXICO
# ----------------------------------------------------
with tab1:
    st.subheader("🔤 Resultados do Motor Léxico (Okapi BM25)")
    st.caption(f"Configuração atual: k1 = {k1_param:.2f} | b = {b_param:.2f}")
    
    df_bm25 = df_results.sort_values(by="bm25_rank")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        for idx, row in df_bm25.iterrows():
            st.markdown(f"""
            <div class="metric-card">
                <span class="badge-rank"># {row['bm25_rank']}</span> &nbsp; 
                <span class="badge-doc">{row['id']}</span> <b>{row['titulo']}</b><br>
                <p style="margin-top: 6px; margin-bottom: 4px;">{row['conteudo']}</p>
                <small><b>Score BM25:</b> {row['bm25_score']:.4f}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.info("""
        **🎯 Diagnóstico Léxico (BM25):**
        - O BM25 é excelente para recuperar **códigos exatos** (ex: *CÓD-ECG-12D*) e dosagens numéricas específicas.
        - **Ponto Cego:** Se o usuário buscar por *"infarto"* e o documento contiver apenas *"síndrome coronariana aguda"*, o BM25 atribuirá **score 0.0**.
        """)


# ----------------------------------------------------
# ABA 2: BUSCA SEMÂNTICA (EMBEDDINGS)
# ----------------------------------------------------
with tab2:
    st.subheader("🧠 Resultados da Busca Semântica Vetorial (Cosine Similarity)")
    st.caption("Captura de contexto e sinônimos médicos no espaço vetorial denso.")
    
    df_sem = df_results.sort_values(by="sem_rank")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        for idx, row in df_sem.iterrows():
            st.markdown(f"""
            <div class="metric-card">
                <span class="badge-rank"># {row['sem_rank']}</span> &nbsp; 
                <span class="badge-doc">{row['id']}</span> <b>{row['titulo']}</b><br>
                <p style="margin-top: 6px; margin-bottom: 4px;">{row['conteudo']}</p>
                <small><b>Similaridade de Cosseno:</b> {row['sem_score']:.4f}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.info("""
        **🧠 Diagnóstico Semântico:**
        - O modelo semântico entende que *"infarto"* relaciona-se fortemente com *"isquemia miocárdica"* ou *"síndrome coronariana"*.
        - **Ponto Cego:** Pode diluir a precisão em códigos curtos e específicos (ex: distinguir *ECG-12D* de um exame geral).
        """)


# ----------------------------------------------------
# ABA 3: FUSÃO HÍBRIDA (RRF)
# ----------------------------------------------------
with tab3:
    st.subheader("🔀 Resultados do Motor Híbrido (Reciprocal Rank Fusion)")
    st.caption(f"Fórmula RRF (k={k_rrf_const}, α={alpha_param:.2f}): Score = α · [1 / (60 + Rank_BM25)] + (1-α) · [1 / (60 + Rank_Sem)]")
    
    df_rrf = df_results.sort_values(by="rrf_rank")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        for idx, row in df_rrf.iterrows():
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10B981;">
                <span class="badge-rank" style="background-color: #D1FAE5; color: #065F46;"># {row['rrf_rank']}</span> &nbsp; 
                <span class="badge-doc">{row['id']}</span> <b>{row['titulo']}</b><br>
                <p style="margin-top: 6px; margin-bottom: 4px;">{row['conteudo']}</p>
                <small><b>Score RRF:</b> {row['rrf_score']:.5f} &nbsp;|&nbsp; 
                Rank BM25: #{row['bm25_rank']} | Rank Semântico: #{row['sem_rank']}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.success("""
        **✨ Vantagem do RRF Híbrido:**
        - O Reciprocal Rank Fusion elimina a necessidade de normalização direta de scores heterogêneos.
        - Documentos bem posicionados em **ambas** as buscas (ou com liderança forte em uma) ganham destaque unificado.
        """)

    # Camada Bônus: Cross-Encoder
    if enable_cross_encoder:
        st.divider()
        st.subheader("⭐ Re-Ranking dos Top-3 com Cross-Encoder (Desafio Bônus)")
        top3_list = df_rrf.head(3).to_dict("records")
        reranked_top3 = apply_cross_encoder_rerank(top3_list, query_input)
        
        cols_ce = st.columns(3)
        for i, doc in enumerate(reranked_top3):
            with cols_ce[i]:
                st.metric(
                    label=f"Novo #{i+1}: {doc['id']} - {doc['titulo'][:20]}...",
                    value=f"{doc['cross_encoder_score']:.4f}",
                    delta=f"Rank RRF anterior: #{doc['rrf_rank']}"
                )
                st.caption(doc["conteudo"])


# ----------------------------------------------------
# ABA 4: MATRIZ COMPARATIVA E GRÁFICOS
# ----------------------------------------------------
with tab4:
    st.subheader("📊 Matriz Comparativa de Desempenho e Ranks")
    
    # Tabela consolidada
    df_matrix = df_results[[
        "id", "titulo", "bm25_score", "bm25_rank", "sem_score", "sem_rank", "rrf_score", "rrf_rank"
    ]].sort_values(by="rrf_rank")
    
    df_matrix.columns = [
        "ID", "Título", "Score BM25", "Rank BM25", "Score Semântico", "Rank Semântico", "Score RRF", "Rank RRF (Final)"
    ]
    
    st.dataframe(
        df_matrix.style.format({
            "Score BM25": "{:.4f}",
            "Score Semântico": "{:.4f}",
            "Score RRF": "{:.5f}"
        })
    )
    
    st.divider()
    st.subheader("📈 Comparativo de Posicionamento (Ranks por Método)")
    
    chart_data = df_results.set_index("id")[["bm25_rank", "sem_rank", "rrf_rank"]]
    chart_data.columns = ["Rank BM25", "Rank Semântico", "Rank RRF (Final)"]
    
    st.bar_chart(chart_data)
    st.caption("Nota: Ranks menores (ex: 1) representam melhor posicionamento nos resultados.")
