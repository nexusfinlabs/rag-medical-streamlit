# rag/preprocessor.py
"""
Text Preprocessing with SpaCy + NLTK for Medical RAG

WHY SpaCy?
----------
SpaCy is the industry standard for NLP pipelines:
- FAST: C-based, processes millions of tokens/sec
- NER: extracts entities (body parts, modalities, drugs, orgs)
- Noun chunks: semantic chunking based on linguistic structure
- Dependency parsing: understands sentence structure

In our medical RAG:
- Extract body parts ("chest", "lung", "liver") from queries
- Extract modality terms ("CT", "MRI", "X-ray")
- Smart chunking of clinical descriptions

WHY NLTK?
---------
NLTK is better for granular preprocessing:
- Sentence splitting (sent_tokenize) — gold standard
- Stopword removal before BM25 (improves precision)
- Stemming: "consolidation" → "consolid" (matches "consolidated")

In our medical RAG:
- Clean metadata text before BM25 indexing
- Split clinical notes into sentences for fine-grained retrieval

Usage:
  from rag.preprocessor import preprocess_query, extract_medical_entities
  
  entities = extract_medical_entities("CT chest lung window 1.25mm SIEMENS")
  # {'body_parts': ['chest', 'lung'], 'modalities': ['CT'], 'manufacturers': ['SIEMENS']}
  
  cleaned = preprocess_query("Show me all the lung CT scans with thin slices")
  # {'original': '...', 'tokens': [...], 'entities': {...}, 'normalized': '...'}
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

# ─── Medical domain knowledge ──────────────────────────────
BODY_PARTS = {
    "chest", "lung", "lobe", "thorax", "thoracic", "pulmonary",
    "liver", "hepatic", "abdomen", "abdominal", "brain", "head",
    "spine", "spinal", "pelvis", "pelvic", "kidney", "renal",
    "heart", "cardiac", "mediastinal", "mediastinum", "bone",
}

MODALITIES = {
    "ct", "mri", "xray", "x-ray", "pet", "ultrasound", "us",
    "mammography", "fluoroscopy", "angiography", "scintigraphy",
}

MANUFACTURERS = {
    "siemens", "ge", "philips", "toshiba", "canon", "hitachi",
    "ge medical systems", "fujifilm",
}

PROTOCOLS = {
    "lung window", "mediastinal window", "bone window", "soft tissue",
    "axial", "coronal", "sagittal", "mip", "mpr",
    "contrast", "non-contrast", "enhanced",
}


# ─── SpaCy-based entity extraction ─────────────────────────
def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text using pattern matching + SpaCy.
    
    For interview: explain that in production you'd train a custom NER model
    on medical annotations (using spacy train), but pattern matching is a
    solid baseline for structured metadata like DICOM.
    """
    text_lower = text.lower()
    
    entities = {
        "body_parts": [],
        "modalities": [],
        "manufacturers": [],
        "protocols": [],
        "measurements": [],
    }
    
    # Body parts
    for bp in BODY_PARTS:
        if bp in text_lower:
            entities["body_parts"].append(bp)
    
    # Modalities
    for mod in MODALITIES:
        if mod in text_lower:
            entities["modalities"].append(mod)
    
    # Manufacturers
    for mfr in MANUFACTURERS:
        if mfr in text_lower:
            entities["manufacturers"].append(mfr)
    
    # Protocols
    for proto in PROTOCOLS:
        if proto in text_lower:
            entities["protocols"].append(proto)
    
    # Measurements (e.g., "1.25mm", "3.0mm", "120kVp")
    measurements = re.findall(r'\d+\.?\d*\s*(?:mm|cm|kvp|kv|ma|mas)', text_lower)
    entities["measurements"] = measurements
    
    return entities


# ─── NLTK-based preprocessing ──────────────────────────────
def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize and clean text for BM25 indexing.
    
    WHY this matters for BM25:
    - Stopwords ("the", "a", "is") add noise to BM25 scores
    - Lowercasing ensures "SIEMENS" matches "siemens"
    - Splitting on delimiters handles metadata format: "key=value | key2=value2"
    """
    # Simple robust tokenizer (no NLTK dependency needed at runtime)
    tokens = re.findall(r'[a-z0-9_]+', text.lower())
    
    # Basic medical stopwords (English stopwords + common non-informative)
    STOP = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after',
        'and', 'but', 'or', 'nor', 'not', 'no', 'so', 'if', 'than',
        'this', 'that', 'these', 'those', 'it', 'its',
    }
    
    return [t for t in tokens if t not in STOP and len(t) > 1]


def preprocess_query(query: str) -> Dict:
    """
    Full query preprocessing pipeline:
    1. Extract medical entities (SpaCy-inspired)
    2. Tokenize for BM25
    3. Generate query variations for better recall
    """
    entities = extract_medical_entities(query)
    tokens = tokenize_for_bm25(query)
    
    # Generate query expansion (synonyms)
    expanded_terms = set(tokens)
    SYNONYMS = {
        "lung": {"pulmonary", "chest", "thoracic"},
        "chest": {"thorax", "thoracic", "lung"},
        "liver": {"hepatic"},
        "brain": {"cerebral", "head"},
        "ct": {"computed tomography", "scan"},
        "thin": {"fine", "1.25", "0.6"},
        "thick": {"3.0", "5.0"},
    }
    for token in tokens:
        if token in SYNONYMS:
            expanded_terms.update(SYNONYMS[token])
    
    return {
        "original": query,
        "tokens": tokens,
        "entities": entities,
        "expanded_tokens": list(expanded_terms),
        "normalized": " ".join(tokens),
    }


# ─── SpaCy NER (full version, requires spacy model) ────────
def extract_entities_spacy(text: str) -> Dict:
    """
    Full SpaCy NER extraction.
    
    Requires: python -m spacy download en_core_web_sm
    
    In interview: explain that you'd use this for:
    - Processing clinical reports (not just metadata)
    - Building entity-aware search filters
    - Chunking medical documents by sentence boundaries
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("[warn] SpaCy not available, using pattern matching fallback")
        return extract_medical_entities(text)
    
    doc = nlp(text)
    
    entities = {
        "named_entities": [(ent.text, ent.label_) for ent in doc.ents],
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
        "sentences": [sent.text.strip() for sent in doc.sents],
    }
    
    # Merge with domain-specific extraction
    medical = extract_medical_entities(text)
    entities.update(medical)
    
    return entities


# ─── scispaCy Medical NER (optional, much better for clinical text) ────
_SCI_NLP = None  # lazy-loaded singleton

def _load_scispacy():
    """Lazy-load scispaCy model (en_ner_bc5cdr_md) once."""
    global _SCI_NLP
    if _SCI_NLP is not None:
        return _SCI_NLP
    try:
        import scispacy  # noqa: F401 — required to register custom components
        import spacy
        _SCI_NLP = spacy.load("en_ner_bc5cdr_md")
        print("[preprocessor] ✅ scispaCy model loaded: en_ner_bc5cdr_md")
        return _SCI_NLP
    except (ImportError, OSError) as e:
        print(f"[preprocessor] ⚠️ scispaCy not available ({e}), using pattern fallback")
        return None


def extract_entities_scispacy(text: str) -> Dict:
    """
    Biomedical NER using scispaCy (Allen AI).

    Model: en_ner_bc5cdr_md — trained on BioCreative V CDR corpus.
    Extracts: DISEASE, CHEMICAL entities from biomedical text.

    Why scispaCy over generic SpaCy?
    - Generic en_core_web_sm: "pneumonia" → no entity detected
    - scispaCy bc5cdr:        "pneumonia" → DISEASE ✅
    - scispaCy also detects drugs, chemicals, syndromes

    Install:
        pip install scispacy
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
    """
    nlp = _load_scispacy()
    if nlp is None:
        return extract_medical_entities(text)

    doc = nlp(text)

    sci_entities = {
        "diseases": [],
        "chemicals": [],
    }
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            sci_entities["diseases"].append(ent.text)
        elif ent.label_ == "CHEMICAL":
            sci_entities["chemicals"].append(ent.text)

    # Merge with domain-specific extraction
    result = extract_medical_entities(text)
    result["diseases"] = sci_entities["diseases"]
    result["chemicals"] = sci_entities["chemicals"]
    result["sci_entities_raw"] = [(ent.text, ent.label_) for ent in doc.ents]

    return result


if __name__ == "__main__":
    # Demo
    queries = [
        "CT chest lung window 1.25mm SIEMENS",
        "Show me liver scans from GE with thin slices",
        "NER1006 clinical trial bowel preparation",
        "Compare axial vs coronal reconstructions",
        "pneumonia consolidation ground glass opacity",
        "adenocarcinoma with methotrexate treatment",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = preprocess_query(q)
        print(f"Tokens: {result['tokens']}")
        print(f"Entities: {result['entities']}")
        print(f"Expanded: {result['expanded_tokens']}")

        # Try scispacy if available
        sci = extract_entities_scispacy(q)
        if "diseases" in sci:
            print(f"scispaCy diseases:  {sci['diseases']}")
            print(f"scispaCy chemicals: {sci['chemicals']}")
