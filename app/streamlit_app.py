import json
import urllib.request
import urllib.error

import streamlit as st
import spacy

from validators.falc_checker import check_text, load_rulebook


OLLAMA_URL = "http://localhost:11434/api/generate"

st.set_page_config(
    page_title="HERMES – FALC Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("HERMES – FALC Simplification Assistant")
st.caption("Prototype: local LLM generation + automatic FALC rule checking")

DEFAULT_TEXT = """La municipalité a décidé de mettre en place un dispositif permettant
aux personnes âgées de bénéficier d'un accompagnement personnalisé
dans leurs démarches administratives."""

if "nlp" not in st.session_state:
    st.session_state.nlp = None
if "book" not in st.session_state or "rules" not in st.session_state:
    st.session_state.book, st.session_state.rules = load_rulebook()
if "simplified" not in st.session_state:
    st.session_state.simplified = ""
if "original_report" not in st.session_state:
    st.session_state.original_report = None
if "simplified_report" not in st.session_state:
    st.session_state.simplified_report = None


def load_nlp():
    if st.session_state.nlp is None:
        st.session_state.nlp = spacy.load("fr_core_news_md")
    return st.session_state.nlp


def simplify_text(text, model):
    prompt = f"""Tu es un assistant spécialisé dans la simplification FALC.

Réécris le texte en français facile à comprendre.

Règles obligatoires :
- Chaque phrase doit contenir au maximum 15 mots.
- Utilise des mots courants et faciles à comprendre.
- Utilise la voix active.
- Présente une seule idée par phrase.
- Écris chaque phrase sur une nouvelle ligne.
- Conserve le temps des verbes du texte original.
- Conserve toutes les informations importantes.
- Explique les mots difficiles avec des mots simples.
- Ne change pas le sens du texte.
- N'ajoute aucune information.
- Vérifie les espaces entre tous les mots.
- Réponds uniquement avec le texte simplifié.

Texte :
{text}
"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["response"].strip()


def display_report(report, heading):
    st.subheader(heading)
    score = report.get("score_percent")
    if score is not None:
        st.metric("Diagnostic score", f"{score}%")
    col1, col2 = st.columns(2)
    col1.metric("Rules checked", report.get("rules_checked", "—"))
    failed = report.get("rules_failed", [])
    col2.metric("Failed rules", len(failed))

    if failed:
        st.warning("Rules with detected violations: " + ", ".join(failed))
    else:
        st.success("No violations detected by the implemented automatic rules.")

    with st.expander("Sentence-level findings", expanded=True):
        any_failure = False
        for sentence in report.get("sentences", []):
            failures = sentence.get("failures", [])
            if failures:
                any_failure = True
                st.markdown(f"**{sentence['sentence_id']} — {sentence['text']}**")
                for item in failures:
                    if isinstance(item, dict):
                        st.write("•", item.get("evidence", str(item)))
                    else:
                        st.write("•", str(item))
        if not any_failure:
            st.write("No sentence-level violations were reported.")

    document_findings = report.get("document_level", [])
    if document_findings:
        with st.expander("Document-level findings"):
            for item in document_findings:
                st.write("•", item.get("evidence", str(item)))

    st.caption(report.get("note", ""))


with st.sidebar:
    st.header("Configuration")
    model = st.selectbox(
        "Local Ollama model",
        ["phi3:mini", "mistral:latest"],
        index=0,
    )
    st.caption("The selected model must already be installed in Ollama.")
    st.markdown("---")
    st.write("**Current prototype**")
    st.write("LLM generation → FALC checker")
    st.write("Phase 1 automatic rules only")

st.subheader("1. Enter the original French text")
original_text = st.text_area(
    "Original text",
    value=DEFAULT_TEXT,
    height=160,
    placeholder="Paste a French paragraph here...",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("Simplify and evaluate", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear generated result"):
        st.session_state.simplified = ""
        st.session_state.original_report = None
        st.session_state.simplified_report = None
        st.rerun()

if run_button:
    if not original_text.strip():
        st.error("Please enter a text first.")
    else:
        try:
            with st.spinner("Loading French NLP model..."):
                nlp = load_nlp()
            with st.spinner(f"Generating text with {model} through Ollama..."):
                generated = simplify_text(original_text, model)
            with st.spinner("Running the FALC checker..."):
                original_report = check_text(
                    original_text, nlp, st.session_state.book, st.session_state.rules
                )
                simplified_report = check_text(
                    generated, nlp, st.session_state.book, st.session_state.rules
                )
            st.session_state.simplified = generated
            st.session_state.original_report = original_report
            st.session_state.simplified_report = simplified_report
            st.success("Simplification and evaluation completed.")
        except urllib.error.URLError as error:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                "and the selected model is installed."
            )
            st.caption(str(error))
        except Exception as error:
            st.error("The operation failed.")
            st.exception(error)

if st.session_state.simplified:
    st.subheader("2. Compare the texts")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original text**")
        st.text_area(
            "Original",
            value=original_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )
    with right:
        st.markdown(f"**Simplified text ({model})**")
        st.text_area(
            "Simplified",
            value=st.session_state.simplified,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    st.subheader("3. Automatic FALC evaluation")
    original_result, simplified_result = st.columns(2)
    with original_result:
        display_report(st.session_state.original_report, "Original text")
    with simplified_result:
        display_report(st.session_state.simplified_report, "Simplified text")

    st.info(
        "The diagnostic score measures only the implemented automatic rules. "
        "It is not an official FALC certification and does not measure "
        "reader comprehension or guarantee preservation of meaning."
    )
else:
    st.caption("Enter a paragraph and select “Simplify and evaluate” to begin.")
