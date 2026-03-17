# app.py
"""
Streamlit Demo — Medical Hybrid RAG

Interactive UI for Hybrid Medical RAG demonstration.
Shows BM25 vs Vector vs Hybrid search comparison.

Run:
  streamlit run app.py
"""
import streamlit as st
import time
import json
from pathlib import Path

st.set_page_config(
    page_title="Medical RAG — Hybrid Search Demo",
    page_icon="🏥",
    layout="wide",
)

# ─── Light Theme CSS ────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #FAFBFC; color: #1a1a2e; }
    .block-container { padding-top: 1rem; }
    
    /* Headers */
    h1 { color: #0f3460; font-size: 2rem; }
    h2 { color: #16213e; border-bottom: 2px solid #e8ecf1; padding-bottom: 6px; }
    h3 { color: #333; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #0f3460; border: none; }
    
    /* Result cards */
    .result-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #238636;
        padding: 1rem;
        margin: 0.6rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .bm25 { border-left-color: #F59E0B; }
    .vector { border-left-color: #6366F1; }
    .hybrid { border-left-color: #238636; }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 8px;
        padding: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    /* Info boxes */
    .info-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1E40AF;
    }
    .help-text {
        color: #6B7280;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────
st.title("🏥 Medical RAG — Hybrid Search")
st.markdown("""
<div class="info-box">
    <strong>¿Qué es esto?</strong> Un buscador inteligente sobre miles de imágenes médicas (TACs, rayos X). 
    Combina búsqueda por <strong>palabras exactas</strong> (BM25) + <strong>significado semántico</strong> (Vector) 
    + <strong>re-evaluación de precisión</strong> (Reranker). Escribe una pregunta y el sistema encuentra las imágenes más relevantes.
</div>
""", unsafe_allow_html=True)

# ─── Sidebar: Architecture + Help ───────────────────────────
with st.sidebar:
    st.header("📖 Guía Rápida")
    
    with st.expander("❓ ¿Cómo funciona?", expanded=False):
        st.markdown("""
        1. **Escribes** una pregunta en lenguaje natural
        2. **BM25** busca por palabras exactas (ej: "SIEMENS")
        3. **Vector** busca por significado (ej: "thin slices" = "1.25mm")
        4. **RRF Fusion** combina ambos rankings
        5. **Reranker** re-evalúa los mejores resultados
        6. **LLaMA** puede razonar sobre lo encontrado
        """)
    
    with st.expander("🔍 ¿Qué modo elegir?", expanded=False):
        st.markdown("""
        - **🟢 Hybrid** (recomendado): combina keyword + semántico
        - **🟡 BM25**: cuando buscas términos exactos (IDs, fabricantes, valores)
        - **🟣 Vector**: cuando buscas por concepto ("thin slices", "lung cancer")
        """)
    
    with st.expander("📝 Queries de ejemplo", expanded=True):
        st.markdown("""
        ```
        lung nodule thin slice
        SIEMENS 1.25mm
        CT THORAX contrast GE
        chest CT screening protocol
        manufacturer kvp 120
        ```
        """)
    
    st.divider()
    st.header("📊 Dataset Cargado")
    
    pairs_path = Path("data/metadata/pairs.jsonl")
    if pairs_path.exists():
        n_docs = sum(1 for _ in open(pairs_path))
        st.metric("CT Slices indexados", f"{n_docs:,}")
    else:
        st.warning("⚠️ pairs.jsonl no encontrado — ejecuta el pipeline de ingesta primero")
        n_docs = 0
    
    st.metric("Fuente", "TCIA LIDC-IDRI")
    st.metric("Modalidad", "CT Chest / Lung")
    
    st.divider()
    st.header("🔧 Stack Técnico")
    st.markdown("""
    | Componente | Modelo |
    |---|---|
    | **Embeddings** | MiniLM-L6-v2 (384d) |
    | **Reranker** | Cross-Encoder MiniLM |
    | **Keywords** | BM25 (rank_bm25) |
    | **LLM** | Ollama / LLaMA3 |
    | **Agent** | LangGraph |
    """)

# ─── Main: Search Interface ────────────────────────────────
st.header("🔍 Buscar en la base de datos médica")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Escribe tu pregunta",
        placeholder="Ej: lung nodule thin slice SIEMENS",
        help="Escribe en inglés. Puedes combinar términos exactos (SIEMENS, 1.25mm) con conceptos (thin slices, lung cancer).",
    )
with col2:
    mode = st.selectbox(
        "Modo de búsqueda",
        ["hybrid", "bm25", "vector"],
        help="🟢 Hybrid = BM25 + Vector combinados (recomendado). 🟡 BM25 = keywords exactos. 🟣 Vector = búsqueda semántica.",
    )

col_k, col_rerank = st.columns(2)
with col_k:
    k = st.slider(
        "Número de resultados (K)",
        3, 20, 10,
        help="K = cuántos resultados quieres ver. K bajo (3-5) para búsquedas específicas, K alto (10-20) para explorar.",
    )
with col_rerank:
    use_reranker = st.checkbox(
        "✨ Activar Reranker",
        value=False,
        help="El Reranker re-evalúa los resultados con un modelo Cross-Encoder más preciso. Tarda ~1-2s extra pero mejora la calidad un 10-20%.",
    )

# ─── Search Execution ──────────────────────────────────────
if query and st.button("🚀 Buscar", type="primary"):
    try:
        from rag.hybrid_search import HybridSearchEngine
        
        with st.spinner("Cargando motor de búsqueda..."):
            engine = HybridSearchEngine()
        
        t0 = time.time()
        results = engine.search(query, k=k, mode=mode, use_reranker=use_reranker)
        elapsed = time.time() - t0
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 Resultados", len(results))
        m2.metric("⏱️ Tiempo", f"{elapsed:.3f}s")
        m3.metric("📈 Top Score", f"{results[0]['score']:.4f}" if results else "N/A")
        mode_labels = {"hybrid": "🟢 Hybrid", "bm25": "🟡 BM25", "vector": "🟣 Vector"}
        m4.metric("🔧 Modo", mode_labels.get(mode, mode))
        
        if use_reranker:
            st.success("✨ Reranker aplicado — resultados re-evaluados con Cross-Encoder")
        
        st.divider()
        
        # Results
        for r in results:
            css_class = mode if mode != "hybrid" else "hybrid"
            text = r.get('text', '')
            # Parse text fields for better display
            parts = text.split(" | ")
            protocol = parts[0] if len(parts) > 0 else ""
            details = " | ".join(parts[1:]) if len(parts) > 1 else ""
            
            st.markdown(f"""
            <div class="result-card {css_class}">
                <strong>#{r['rank']}</strong> &nbsp; 
                score: <code>{r['score']:.4f}</code> &nbsp; | &nbsp;
                doc: <code>{r.get('doc_id', 'N/A')[:40]}...</code><br/>
                <strong>{protocol}</strong><br/>
                <small style="color:#666;">{details}</small><br/>
                <small>📁 {r.get('dicom_path', r.get('image_path', 'N/A'))}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # ─── Explanatory Box ──────────────────────────────────────
        st.divider()
        st.subheader("📖 ¿Cómo interpretar estos resultados?")
        
        # Analyze what's in the results
        if results:
            first = results[0].get('text', '')
            explanations = []
            
            if "W/CONTRAST" in first or "CONTRAST" in first.upper():
                explanations.append("💉 **Con contraste (W/CONTRAST):** Se inyectó contraste intravenoso. Resalta tumores y vasos — el tumor capta más contraste que el tejido sano.")
            if "WO CONTRAST" in first.upper() or "WITHOUT" in first.upper():
                explanations.append("🚫 **Sin contraste (WO):** Estudio de rutina o seguimiento. Menos invasivo, útil para comparar con estudios previos.")
            
            # Thickness
            import re
            thick_match = re.search(r'thickness=(\d+\.?\d*)', first)
            if thick_match:
                thick_val = float(thick_match.group(1))
                if thick_val <= 1.5:
                    explanations.append(f"📏 **Thickness={thick_val}mm (FINO):** Cortes finos — detecta nódulos pequeños desde 3mm. Protocolo de alta resolución.")
                elif thick_val <= 3.0:
                    explanations.append(f"📏 **Thickness={thick_val}mm (ESTÁNDAR):** Cortes estándar — buen equilibrio entre resolución y dosis de radiación.")
                else:
                    explanations.append(f"📏 **Thickness={thick_val}mm (GRUESO):** Cortes gruesos — menos resolución pero menor dosis. Típico de screening o seguimiento.")
            
            # kVp
            kvp_match = re.search(r'kvp=(\d+)', first)
            if kvp_match:
                kvp_val = int(kvp_match.group(1))
                if kvp_val >= 120:
                    explanations.append(f"⚡ **kVp={kvp_val} (ESTÁNDAR):** Voltaje estándar adulto. Buena calidad de imagen.")
                else:
                    explanations.append(f"⚡ **kVp={kvp_val} (LOW-DOSE):** Voltaje reducido = {int((1-kvp_val/120)*100)}% menos radiación. Protocolo low-dose (pediátrico/screening).")
            
            # Manufacturer
            if "GE" in first:
                explanations.append("🏭 **GE Medical Systems:** Fabricante americano. Sus protocolos suelen usar reconstrucción ASiR.")
            elif "SIEMENS" in first:
                explanations.append("🏭 **Siemens Healthineers:** Fabricante alemán. Conocido por protocolos CARE Dose 4D (optimización automática de dosis).")
            elif "PHILIPS" in first:
                explanations.append("🏭 **Philips Healthcare:** Fabricante holandés. Fuerte en iDose (reconstrucción iterativa).")
            
            # Score interpretation
            top_score = results[0]['score']
            if mode == "bm25":
                explanations.append(f"📊 **Score BM25={top_score:.4f}:** Score alto = muchas keywords coinciden. BM25 > 10 es generalmente buena coincidencia.")
            elif mode == "vector":
                explanations.append(f"📊 **Score Vector={top_score:.4f}:** Cosine similarity. Score > 0.3 = buena coincidencia semántica. Score > 0.5 = excelente.")
            else:
                explanations.append(f"📊 **Score Hybrid (RRF)={top_score:.4f}:** Score combinado BM25+Vector. No tiene unidad — solo comparar entre resultados de la misma búsqueda.")
            
            if len(results) == 0 or not explanations:
                st.info("No hay suficiente metadata para interpretar. Los resultados muestran la información disponible del DICOM.")
            else:
                for exp in explanations:
                    st.markdown(exp)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Asegúrate de tener `data/metadata/pairs.jsonl` y `data/embeddings/text_embeddings.npz`")

# ─── Comparison Mode ────────────────────────────────────────
st.divider()
st.header("⚔️ Comparador — BM25 vs Vector vs Hybrid")

st.markdown("""
<div class="info-box">
    <strong>¿Para qué sirve?</strong> Escribe UNA query y compara los resultados de los 3 métodos lado a lado. 
    Así puedes ver por qué <strong>Hybrid</strong> es mejor: combina la precisión de BM25 con el entendimiento de Vector.
</div>
""", unsafe_allow_html=True)

st.markdown("""
| Query de ejemplo | Mejor modo | Por qué |
|---|---|---|
| `NER1006 SIEMENS` | 🟡 BM25 | Términos exactos — BM25 encuentra keyword matches |
| `lung scans with thin slices` | 🟣 Vector | Significado semántico — entiende sinónimos |
| `Compare GE vs SIEMENS 1.25mm` | 🟢 Hybrid | Necesita ambos: marcas exactas + contexto semántico |
| `what protocols are common` | 🟣 Vector | Sin términos exactos, necesita entendimiento semántico |
""")

compare_query = st.text_input(
    "Escribe una query para comparar los 3 modos:",
    placeholder="Ej: SIEMENS 1.25mm lung",
    help="Escribe la misma query y verás cómo cada método devuelve resultados diferentes.",
)

if compare_query and st.button("⚔️ Comparar los 3 modos"):
    try:
        from rag.hybrid_search import HybridSearchEngine
        
        with st.spinner("Cargando motor..."):
            engine = HybridSearchEngine()
        
        c1, c2, c3 = st.columns(3)
        
        for col, m, label in [(c1, "bm25", "🟡 BM25 (Keywords)"), (c2, "vector", "🟣 Vector (Semántico)"), (c3, "hybrid", "🟢 Hybrid (Combinado)")]:
            with col:
                st.subheader(label)
                t0 = time.time()
                res = engine.search(compare_query, k=5, mode=m)
                elapsed = time.time() - t0
                st.caption(f"⏱️ {elapsed:.3f}s | {len(res)} resultados")
                for r in res:
                    st.markdown(f"**#{r['rank']}** score=`{r['score']:.4f}`")
                    text_short = r.get('text', '')[:60]
                    st.caption(text_short)
                if not res:
                    st.warning("Sin resultados")
    except Exception as e:
        st.error(str(e))

# ─── Footer ─────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.85rem; padding:1rem;">
    🏥 Medical Hybrid RAG v2.0 | TCIA LIDC-IDRI Dataset | 
    Built with HuggingFace, BM25, FAISS, LangGraph, Ollama
</div>
""", unsafe_allow_html=True)
