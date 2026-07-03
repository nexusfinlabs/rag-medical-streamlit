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

    /* RBAC access badges */
    .access-ok {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.6rem 0 1rem;
        font-size: 0.9rem;
        color: #065F46;
    }
    .access-denied {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #DC2626;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0 1rem;
        font-size: 0.92rem;
        color: #7F1D1D;
    }
    .access-denied strong { color: #991B1B; }
    .access-denied table { width: 100%; border-collapse: collapse; margin-top: 0.6rem; font-size: 0.85rem; }
    .access-denied td, .access-denied th { padding: 0.3rem 0.5rem; text-align: left; border-bottom: 1px solid #FCA5A5; }
    .access-denied td.yes { color: #065F46; font-weight: 600; }
    .access-denied td.no { color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# ─── RBAC: rol -> acceso a este datalake ────────────────────
ROLES = ["derma", "clintrials", "pv", "commercial", "regulatory", "market-access", "medinfo", "biostat"]
ROLE_LABELS = {
    "derma": "Dermatology", "clintrials": "Clinical Operations", "pv": "Pharmacovigilance",
    "commercial": "Commercial", "regulatory": "Regulatory Affairs", "market-access": "Market Access",
    "medinfo": "Medical Affairs", "biostat": "Biostatistics",
}
ROLE_DATALAKE = {
    "derma": "rag-derma-datalake3", "clintrials": "rag-clintrials-datalake1", "pv": "rag-pv-datalake4",
    "commercial": "rag-commercial-datalake2", "regulatory": "rag-regulatory-datalake5",
    "market-access": "rag-market-access-datalake6", "medinfo": "rag-medinfo-datalake7", "biostat": "rag-clintrials-datalake1",
}
AUTHORIZED_ROLES = {"derma", "clintrials"}
THIS_DATALAKE = "rag-derma-datalake3 (CT + Dermatología combinado)"

# ─── FinOps: precio input/output por 1M tokens ──────────────
MODEL_PRICING = {
    "Claude Opus 4.8":  {"input": 5.00, "output": 25.00, "note": "precio oficial Anthropic"},
    "Claude Sonnet 5":  {"input": 3.00, "output": 15.00, "note": "precio oficial Anthropic (intro $2.00/$10.00 hasta 31-ago-2026)"},
    "GPT-5":            {"input": 5.00, "output": 15.00, "note": "estimado — verificar con OpenAI"},
    "Gemini 2.5 Pro":   {"input": 1.25, "output": 5.00,  "note": "estimado — verificar con Google"},
}
ROLE_MODEL = {
    "pv": "Claude Opus 4.8", "regulatory": "Claude Opus 4.8", "biostat": "Claude Opus 4.8",
    "derma": "Claude Sonnet 5", "clintrials": "Claude Sonnet 5", "commercial": "Claude Sonnet 5",
    "market-access": "Claude Sonnet 5", "medinfo": "Claude Sonnet 5",
}

def estimar_tokens_billing(query, results, model, output_tokens_est=400):
    chars = len(query) + sum(len(r.get("text", "")) for r in results)
    input_tokens = max(1, chars // 4)
    price = MODEL_PRICING[model]
    coste = (input_tokens / 1_000_000 * price["input"]) + (output_tokens_est / 1_000_000 * price["output"])
    return input_tokens, output_tokens_est, coste, price

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
        st.markdown("**🫁 Cáncer de pulmón (CT):**")
        lung_queries = [
            "lung nodule thin slice SIEMENS 1.25mm",
            "CT THORAX contrast GE chest screening kvp 120",
            "pulmonary nodule spiculated margin low-dose",
        ]
        for q in lung_queries:
            if st.button(q, key=f"btn_lung_{q[:20]}", use_container_width=True):
                st.session_state["query_prefill"] = q

        st.markdown("**🩺 Dermatología (Fitzpatrick17k):**")
        derma_queries = [
            "atopic dermatitis eczema pruritus inflammatory",
            "psoriasis plaque erythematous scaling skin",
            "melanoma suspicious lesion asymmetry border",
        ]
        for q in derma_queries:
            if st.button(q, key=f"btn_derma_{q[:20]}", use_container_width=True):
                st.session_state["query_prefill"] = q

    st.divider()
    st.header("📊 Datasets Indexados")

    ct_path = Path("data/metadata/pairs.jsonl")
    derma_path = Path("/data/rag-derma/data/metadata/pairs.jsonl")

    if ct_path.exists():
        n_ct = sum(1 for _ in open(ct_path))
        st.metric("🫁 CT Slices (LIDC-IDRI)", f"{n_ct:,}")
    else:
        n_ct = 0

    if derma_path.exists():
        n_derma = sum(1 for _ in open(derma_path))
        st.metric("🩺 Casos Derma (Fitzpatrick17k)", f"{n_derma:,}")
    else:
        st.info("⏳ Indexando Fitzpatrick17k...")
        n_derma = 0

    st.metric("Fuente CT", "TCIA LIDC-IDRI")
    st.metric("Fuente Derma", "Fitzpatrick17k (114 diagnósticos)")
    
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
st.header("🤖 Prompt Datos Médicos (Claude Chat/Code, GPT Chat/Codex)")
st.markdown("""
<div class="info-box">
    Esta interfaz es equivalente a ejecutar el mismo prompt directamente en <strong>Claude Desktop</strong> o en <strong>Claude.ai (Chat)</strong> — el conector MCP con RBAC por rol es el mismo, así que lo que ves aquí (acceso autorizado o no, según el rol) es exactamente lo que verías allí.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    prefill = st.session_state.pop("query_prefill", "")
    query = st.text_input(
        "Escribe tu pregunta",
        value=prefill,
        placeholder="Ej: atopic dermatitis eczema | lung nodule thin slice SIEMENS",
        help="Escribe en inglés. Usa los botones de la barra lateral para queries de ejemplo.",
    )
with col2:
    mode = st.selectbox(
        "Modo de búsqueda",
        ["hybrid", "bm25", "vector"],
        help="🟢 Hybrid = BM25 + Vector combinados (recomendado). 🟡 BM25 = keywords exactos. 🟣 Vector = búsqueda semántica.",
    )

col_model, col_role = st.columns(2)
with col_model:
    model = st.selectbox(
        "Modelo IA",
        ["Claude Opus 4.8", "Claude Sonnet 5", "GPT-5", "Gemini 2.5 Pro"],
        help="Modelo que razona sobre los resultados recuperados. Claude se conecta vía MCP con RBAC por rol.",
    )
with col_role:
    role = st.selectbox(
        "Rol",
        ROLES,
        format_func=lambda r: f"{r} — {ROLE_LABELS[r]}",
        help="Simula con qué rol/squad entras a Claude. Determina si tienes acceso a este datalake concreto.",
    )

is_authorized = role in AUTHORIZED_ROLES
if is_authorized:
    st.markdown(f"""
    <div class="access-ok">✅ <strong>Acceso autorizado</strong> — el rol <strong>{role} ({ROLE_LABELS[role]})</strong> tiene permiso RBAC sobre <strong>{THIS_DATALAKE}</strong>.</div>
    """, unsafe_allow_html=True)
else:
    rows = "".join(
        f"<tr><td>{r} — {ROLE_LABELS[r]}</td><td class='{'yes' if r in AUTHORIZED_ROLES else 'no'}'>{'✅ ' + ROLE_DATALAKE[r] if r in AUTHORIZED_ROLES else '🚫 sin acceso a ' + THIS_DATALAKE.split(' ')[0]}</td></tr>"
        for r in ROLES
    )
    st.markdown(f"""
    <div class="access-denied">
        🚫 <strong>Acceso No autorizado</strong> — el rol <strong>{role} ({ROLE_LABELS[role]})</strong> no tiene permiso RBAC sobre <strong>{THIS_DATALAKE}</strong>.
        Solo <strong>Dermatology</strong> y <strong>Clinical Operations</strong> tienen acceso a este datalake concreto — cada squad ve únicamente los conectores MCP de su ámbito, igual que en Claude Desktop.
        <table>
            <tr><th>Rol</th><th>Acceso a este datalake</th></tr>
            {rows}
        </table>
    </div>
    """, unsafe_allow_html=True)

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
if query and st.button("🚀 Buscar", type="primary", disabled=not is_authorized):
    if not is_authorized:
        st.stop()
    try:
        from rag.hybrid_search import HybridSearchEngine
        
        with st.spinner("Cargando motor de búsqueda..."):
            engine = HybridSearchEngine(pairs_path='/data/rag-combined/data/metadata/pairs.jsonl', text_emb_path='/data/rag-combined/data/embeddings/text_embeddings.npz')
        
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

        # ─── FinOps: tokens y billing en tiempo real ───────────
        st.divider()
        st.subheader("💰 Calcula tokens/billing")
        in_tok, out_tok, coste, price = estimar_tokens_billing(query, results, model)
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("📥 Tokens input (aprox)", f"{in_tok:,}")
        f2.metric("📤 Tokens output (est)", f"{out_tok:,}")
        f3.metric("💵 Coste este query", f"${coste:.5f}")
        f4.metric("🏷️ Modelo", model)
        st.caption(
            f"Estimación ~4 caracteres/token sobre la pregunta + el texto recuperado de **{THIS_DATALAKE}**. "
            f"Precio {model}: ${price['input']:.2f} / ${price['output']:.2f} por 1M tokens (input/output) — {price['note']}. "
            f"Se facturaría internamente contra el squad **{role} ({ROLE_LABELS[role]})** vía **{ROLE_DATALAKE[role]}**, "
            f"agregado en `finops/usage_by_role.csv` (ver Fase 2 · Gobierno operativo del roadmap)."
        )

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
            engine = HybridSearchEngine(pairs_path='/data/rag-combined/data/metadata/pairs.jsonl', text_emb_path='/data/rag-combined/data/embeddings/text_embeddings.npz')
        
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

# ─── Squads → Datalakes → Skills → MCP ─────────────────────
st.divider()
st.header("🗂️ Squads → Datalakes → Skills → MCP")
st.markdown("""
| Squad (rol) | Datalake | MCP servers | Skills por defecto | Modelo propuesto |
|---|---|---|---|---|
| derma | rag-derma-datalake3 | rag-derma, rag-clintrials, rag-medinfo, rag-competitive-intel | scoring-easi-pasi-imagen, benchmarking-competitivo-derma | Claude Sonnet 5 |
| clintrials (ClinOps) | rag-clintrials-datalake1 | rag-clintrials, rag-pv, rag-competitive-intel, ctms-edc | feasibility-y-reclutamiento, desviaciones-protocolo | Claude Sonnet 5 |
| commercial | rag-commercial-datalake2 | rag-commercial, rag-medinfo, rag-kol, veeva-vault | validacion-reclamos-mlr, argumentario-y-contenido | Claude Sonnet 5 |
| pv | rag-pv-datalake4 | rag-pv, rag-medinfo, rag-regulatory, argus-safety | busqueda-senal, redaccion-psur, intake-triage-icsr | Claude Opus 4.8 |
| regulatory | rag-regulatory-datalake5 | rag-regulatory, rag-medinfo, rag-clintrials, veeva-vault | ensamblaje-ectd, respuesta-preguntas-agencia | Claude Opus 4.8 |
| market-access | rag-market-access-datalake6 | rag-market-access, rag-clintrials, rag-medinfo | dossier-hta, gap-analysis-evidencia | Claude Sonnet 5 |
| medinfo | rag-medinfo-datalake7 | rag-medinfo, rag-pv, rag-regulatory, rag-kol | respuesta-consulta-medica, alineacion-ficha-tecnica | Claude Sonnet 5 |
| biostat | rag-clintrials-datalake1 (vía Claude Code) | rag-clintrials, rag-regulatory, databricks-unity | revision-sap-plan-analisis, validacion-tlf-adam | Claude Opus 4.8 |
""")
st.caption(
    "Datos verificados contra los 8 `plugin.json` y `.mcp.json` reales del repo pharma-plugins. "
    "Modelo propuesto por criticidad — Opus 4.8 en squads de alto riesgo (PV, Regulatory, Biostat), "
    "Sonnet 5 en el resto por coste/latencia — a confirmar antes de fijarlo en GOBERNANZA.md."
)

# ─── FinOps: simulación de coste/billing por squad ─────────
st.divider()
st.header("💰 FinOps — simulación de coste/billing por squad")

periodo = st.radio("Periodo de facturación", ["Semana", "Mes"], horizontal=True, key="finops_periodo")
multiplicador = 1 if periodo == "Semana" else 4

def format_tokens(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

SQUAD_USAGE_SIM = {
    "pv":            {"queries_semana": 120, "tok_in": 3500, "tok_out": 600},
    "commercial":    {"queries_semana": 400, "tok_in": 1800, "tok_out": 400},
    "regulatory":    {"queries_semana": 90,  "tok_in": 5000, "tok_out": 700},
    "clintrials":    {"queries_semana": 150, "tok_in": 2800, "tok_out": 500},
    "market-access": {"queries_semana": 60,  "tok_in": 2200, "tok_out": 450},
    "medinfo":       {"queries_semana": 250, "tok_in": 2000, "tok_out": 400},
    "derma":         {"queries_semana": 100, "tok_in": 2600, "tok_out": 450},
    "biostat":       {"queries_semana": 40,  "tok_in": 4000, "tok_out": 600},
}

rows, model_agg = [], {}
total_cost = total_tokens = 0

for squad, usage in SQUAD_USAGE_SIM.items():
    queries = usage["queries_semana"] * multiplicador
    tok_in = queries * usage["tok_in"]
    tok_out = queries * usage["tok_out"]
    model = ROLE_MODEL[squad]
    price = MODEL_PRICING[model]
    cost = (tok_in / 1_000_000 * price["input"]) + (tok_out / 1_000_000 * price["output"])
    total_cost += cost
    total_tokens += tok_in + tok_out
    agg = model_agg.setdefault(model, {"tokens": 0, "cost": 0.0, "squads": 0})
    agg["tokens"] += tok_in + tok_out
    agg["cost"] += cost
    agg["squads"] += 1
    rows.append((squad, model, ROLE_DATALAKE[squad], queries, tok_in, tok_out, cost))

st.subheader("Por squad / equipo")
table_md = "| Squad | Modelo | Datalake | Queries | Tokens in | Tokens out | Coste |\n|---|---|---|---|---|---|---|\n"
for squad, model, datalake, queries, tok_in, tok_out, cost in rows:
    table_md += f"| {squad} — {ROLE_LABELS[squad]} | {model} | {datalake} | {queries:,} | {format_tokens(tok_in)} | {format_tokens(tok_out)} | ${cost:,.2f} |\n"
st.markdown(table_md)

fc1, fc2, fc3 = st.columns(3)
fc1.metric(f"💵 Coste total org / {periodo.lower()}", f"${total_cost:,.2f}")
fc2.metric(f"🔢 Tokens totales / {periodo.lower()}", format_tokens(total_tokens))
fc3.metric("👥 Squads activos", len(SQUAD_USAGE_SIM))

st.subheader("Agregado por modelo")
model_md = "| Modelo | Squads que lo usan | Tokens totales | Coste total |\n|---|---|---|---|\n"
for model, agg in model_agg.items():
    model_md += f"| {model} | {agg['squads']} | {format_tokens(agg['tokens'])} | ${agg['cost']:,.2f} |\n"
st.markdown(model_md)

st.caption(
    "Simulación ilustrativa — volúmenes de queries y tokens estimados para dar ejemplo, no son datos reales de producción. "
    "En producción esto vendría de `finops/usage_by_role.csv` (export periódico de la Admin Console de Claude Enterprise "
    "cruzado con pertenencia a grupos de Entra ID) y se agregaría en `finops/showback_by_department.xlsx` para el steering."
)

# ─── Footer ─────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.85rem; padding:1rem;">
    🏥 Medical Hybrid RAG v2.0 | TCIA LIDC-IDRI Dataset | 
    Built with HuggingFace, BM25, FAISS, LangGraph, Ollama
</div>
""", unsafe_allow_html=True)
