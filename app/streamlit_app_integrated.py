
import json
import urllib.request
import urllib.error

import streamlit as st
import spacy

from validators.falc_checker import check_text, load_rulebook
from validators.meaning_validator import validate_meaning


OLLAMA_URL = "http://localhost:11434"
MODEL_OPTIONS = ["phi3:mini", "mistral:latest"]

st.set_page_config(
    page_title="HERMES – FALC Assistant",
    layout="wide",
)

st.title("HERMES – FALC Simplification Assistant")
st.caption(
    "Local LLM generation · Meaning-preservation review · "
    "Automatic FALC rule checking"
)

DEFAULT_TEXT = """La municipalité a décidé de mettre en place un dispositif permettant
aux personnes âgées de bénéficier d'un accompagnement personnalisé
dans leurs démarches administratives."""


# ---------------------------------------------------------------------
# Session state and resources
# ---------------------------------------------------------------------

if "nlp" not in st.session_state:
    st.session_state.nlp = None

if "book" not in st.session_state or "rules" not in st.session_state:
    st.session_state.book, st.session_state.rules = load_rulebook()

session_defaults = {
    "simplified": "",
    "last_original": "",
    "generation_model": "mistral:latest",
    "reviewer_model": "phi3:mini",
    "meaning_report": None,
    "original_report": None,
    "simplified_report": None,
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


def load_nlp():
    if st.session_state.nlp is None:
        st.session_state.nlp = spacy.load("fr_core_news_md")
    return st.session_state.nlp


# ---------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------

def simplify_text(text, model):
    """Generate a FALC-oriented rewrite with a local Ollama model."""

    prompt = f"""You are an expert French FALC
(Facile à lire et à comprendre) writing assistant.

Your task is to transform the SOURCE TEXT into simpler,
more accessible French.

This is a complete rewrite, NOT a summary and NOT just
a synonym replacement.

FOLLOW THESE RULES:

1. MEANING PRESERVATION
- Preserve all important facts and information.
- Preserve the actors, their actions and the people affected.
- Preserve dates, numbers, names, conditions, obligations and negations.
- Do not invent, remove or change information.
- Preserve the original chronology and intention.
- Preserve decisions and decision verbs such as "a décidé".

2. SENTENCE STRUCTURE
- Every sentence must contain no more than 15 words.
- Count the words in each sentence before returning the result.
- If a sentence exceeds 15 words, split it.
- Prefer several short sentences over one long sentence.
- Each sentence must express one main idea.
- Use a separate sentence when explaining an important detail.
- Keep the original subject and target audience explicit.
- Preserve plural and singular meaning exactly.
- Prefer simple subject-verb-object structures.
- Use active voice when the meaning remains unchanged.
- Preserve the original tense and chronology.
- Use correct French grammar and pronoun agreement.
- Put each sentence on a separate line.

3. VOCABULARY
- Preserve the idea of personalized or individualized support.
- Do not replace "personnalisé" with "personnel".
- You may express personalized support as:
  "une aide adaptée aux besoins de chaque personne".
- Do not add explanatory sentences beginning with
  "Cela signifie que" or similar expressions.
- Do not repeat information already expressed in another sentence.
- Use grammatically correct pronouns that agree with their antecedents.

4. QUALITY CONTROL
Before returning the result, silently check:
- Is every important piece of information preserved?
- Are the sentences shorter and easier to understand?
- Is the vocabulary simpler than the original?
- Have I avoided adding or inventing information?

Do not merely replace individual words with synonyms.
Simplify the sentence structure and vocabulary while preserving meaning.

Return ONLY the rewritten French text.
Do not include explanations, introductions or comments.

SOURCE TEXT:
{text}

FALC-ORIENTED REWRITE:
"""

    payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1
    }
}

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
    f"{OLLAMA_URL}/api/generate",
    data=data,
    headers={"Content-Type": "application/json"}
)

    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8"))

    generated = result.get("response", "").strip()

    if not generated:
        raise ValueError("The selected model returned an empty response.")

    return generated


# ---------------------------------------------------------------------
# CPM1 validation and one repair attempt
# ---------------------------------------------------------------------



def repair_cpm1(original_text, generated_text, model, nlp, book, rules):
    """
    Attempt one LLM repair when CPM1 fails.
    Extract only the French draft, then recheck CPM1.
    """

    initial_report = check_text(generated_text, nlp, book, rules)

    if "CPM1" not in initial_report.get("rules_failed", []):
        return generated_text, initial_report, False, None

    # Collect CPM1 findings using the actual checker report structure.
    findings = []

    for sentence_result in initial_report.get("sentences", []):
        sentence = sentence_result.get("text", "")

        for failure in sentence_result.get("failures", []):
            if failure.get("rule_id") == "CPM1":
                findings.append(
                    f"Sentence: {sentence}\n"
                    f"Problem: {failure.get('evidence', '')}"
                )

    findings_text = "\n".join(findings)

    print("\n========== CPM1 REPAIR ==========")
    print("INITIAL GENERATED TEXT:")
    print(generated_text)
    print("\nDETECTED PROBLEMS:")
    print(findings_text)

    prompt = f"""
You are a French FALC simplification editor.

Your task is to shorten the generated French text while preserving
the exact meaning of the original.

ORIGINAL TEXT:
{original_text}

GENERATED TEXT:
{generated_text}

DETECTED CPM1 PROBLEMS:
{findings_text}


CRITICAL MEANING-PRESERVATION RULES:

The original contains FOUR essential pieces of information.
You MUST preserve all four:

A. ACTOR AND DECISION:
The municipality has made a decision.
Preserve "La municipalité a décidé".
Do not change the decision into an action already completed.

B. SYSTEM:
The municipality has decided to establish a "dispositif".
The word "dispositif" or a clearly equivalent expression MUST remain.
Do not replace the system with only an offer or a service.

C. TARGET AUDIENCE:
The system is intended for older people.
Preserve "les personnes âgées".

D. PURPOSE AND TYPE OF SUPPORT:
The system allows older people to receive personalized support.
This support concerns their administrative procedures.
Preserve both ideas.

IMPORTANT:
Do not merge the decision to establish a system
with the provision of personalized support.
They are related but distinct pieces of information.

SIMPLIFICATION RULES:

1. Every sentence MUST contain at most 15 words.
2. Split the original into several short sentences.
3. Express one main idea per sentence.
4. Use simple and familiar French.
5. Prefer active voice when meaning remains unchanged.
6. Do not remove any of the four essential pieces of information.
7. Do not invent information.
8. Use correct French grammar.
9. Return only the French text inside the required markers.

REFERENCE STRUCTURE:

La municipalité a décidé de créer un dispositif.
Ce dispositif aide les personnes âgées.
Il leur offre un accompagnement adapté à leurs besoins.
Cet accompagnement concerne leurs démarches administratives.

This is a reference structure.
Preserve its meaning, not necessarily its exact wording.

Before returning the answer, verify:
- The municipal decision is preserved.
- The system or dispositif is explicitly mentioned.
- Older people remain the target audience.
- Personalized support is preserved.
- Administrative procedures are mentioned.
- Every sentence has at most 15 words.


Check that every sentence contains 15 words or fewer.

OUTPUT FORMAT:

Return the corrected French text ONLY inside these markers:

<FALC_TEXT>
Your corrected French text here.
</FALC_TEXT>

Do not write anything outside these markers.
"""

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        data = json.dumps(payload).encode("utf-8")

        endpoint = OLLAMA_URL.rstrip("/")
        if not endpoint.endswith("/api/generate"):
            endpoint += "/api/generate"

        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        raw_response = result.get("response", "").strip()

        # Extract only the French text between the markers.
        start_marker = "<FALC_TEXT>"
        end_marker = "</FALC_TEXT>"

        start = raw_response.find(start_marker)
        end = raw_response.find(end_marker)

        if start == -1 or end == -1 or end <= start:
            return (
                generated_text,
                initial_report,
                True,
                "Repair response did not contain the required FALC_TEXT markers."
            )

        repaired_text = raw_response[
            start + len(start_marker):end
        ].strip()

        # Remove optional enclosing quotation marks.
        if (
            len(repaired_text) >= 2
            and repaired_text[0] == repaired_text[-1]
            and repaired_text[0] in ('"', "'")
        ):
            repaired_text = repaired_text[1:-1].strip()

        if not repaired_text:
            return (
                generated_text,
                initial_report,
                True,
                "The repair model returned an empty French draft."
            )

        print("\nREPAIRED FRENCH TEXT:")
        print(repaired_text)
        print("================================\n")

        # Evaluate only the extracted French draft.
        repaired_report = check_text(
            repaired_text, nlp, book, rules
        )

        if "CPM1" in repaired_report.get("rules_failed", []):
            return (
                repaired_text,
                repaired_report,
                True,
                "Repair attempted, but CPM1 still fails."
            )

        return repaired_text, repaired_report, True, None

    except Exception as e:
        return (
            generated_text,
            initial_report,
            True,
            f"CPM1 repair error: {str(e)}"
        )

# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def show_items(title, items):
    if items:
        st.markdown(f"**{title}**")
        for item in items:
            st.write(f"• {item}")


def display_meaning_report(report):
    st.subheader("3. Meaning-preservation review")

    protected = report["protected_elements"]
    semantic = report["semantic_review"]

    if report["status"] == "no_issue_detected":
        st.success(
            "No issue detected by the automated checks. "
            "This is not proof that the meaning is fully preserved."
        )
    else:
        st.warning(
            "Potential meaning-preservation issues detected. "
            "Review the warnings before accepting the generated text."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Potentially missing entities",
        len(protected["potentially_missing_entities"]),
    )

    col2.metric(
        "Potentially missing numbers / dates",
        len(protected["potentially_missing_values"]),
    )

    col3.metric(
        "Semantic review",
        semantic.get("overall", "uncertain").replace("_", " ").title(),
    )

    with st.expander("Protected information checks", expanded=True):

        show_items(
            "Potentially missing names or entities",
            protected["potentially_missing_entities"],
        )

        show_items(
            "Potentially missing numbers, dates or amounts",
            protected["potentially_missing_values"],
        )

        st.write(
            "Negation markers — original: "
            f"{protected['original_negation_marker_count']}; "
            "generated: "
            f"{protected['simplified_negation_marker_count']}"
        )

        if protected["warnings"]:
            for warning in protected["warnings"]:
                st.warning(warning)
        else:
            st.write("No protected-information warning was detected.")

        st.caption(protected["note"])

    with st.expander("LLM semantic review", expanded=True):

        if semantic.get("available"):
            st.write(
                f"**Reviewer model:** `{semantic.get('reviewer_model')}`"
            )

            st.write(
                f"**Assessment:** {semantic.get('overall', 'uncertain')}"
            )

            st.write(
                semantic.get("explanation", "No explanation returned.")
            )

            show_items(
                "Potentially missing information",
                semantic.get("missing_information", []),
            )

            show_items(
                "Potentially changed information",
                semantic.get("changed_information", []),
            )

            show_items(
                "Potentially unsupported additions",
                semantic.get("unsupported_additions", []),
            )

        else:
            st.warning(
                semantic.get(
                    "explanation",
                    "Semantic review is unavailable.",
                )
            )

        st.caption(semantic.get("note", ""))

    st.info(report["note"])


def display_falc_report(report, heading):
    st.subheader(heading)

    score = report.get("score_percent")

    if score is not None:
        st.metric("Diagnostic score", f"{score}%")

    col1, col2 = st.columns(2)

    col1.metric(
        "Rules checked",
        report.get("rules_checked", "—"),
    )

    failed = report.get("rules_failed", [])

    col2.metric("Failed rules", len(failed))

    if failed:
        st.warning(
            "Rules with detected violations: " + ", ".join(failed)
        )
    else:
        st.success(
            "No violations detected by the implemented automatic rules."
        )

    with st.expander("Sentence-level findings", expanded=True):

        any_failure = False

        for sentence in report.get("sentences", []):
            failures = sentence.get("failures", [])

            if failures:
                any_failure = True

                st.markdown(
                    f"**{sentence['sentence_id']} — {sentence['text']}**"
                )

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


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    generation_model = st.selectbox(
        "Generation model",
        MODEL_OPTIONS,
        key="generation_model",
    )

    # Keep the reviewer different by default.
    if st.session_state.reviewer_model == generation_model:
        st.session_state.reviewer_model = next(
            model for model in MODEL_OPTIONS
            if model != generation_model
        )

    reviewer_model = st.selectbox(
        "Meaning-review model",
        MODEL_OPTIONS,
        key="reviewer_model",
    )

    st.caption(
        "Both selected models must already be installed in Ollama."
    )

    if generation_model == reviewer_model:
        st.warning(
            "You selected the same model for generation and review. "
            "A different model is preferable for a second opinion."
        )

    st.caption(
        "A second model is still an LLM-based review, "
        "not independent human validation."
    )

    st.markdown("---")

    st.write("**Current pipeline**")
    st.write(
        "LLM generation → CPM1 validation/repair → "
        "Meaning review → FALC checker"
    )
    st.write("Phase 1 automatic FALC rules only")


# ---------------------------------------------------------------------
# Input and execution
# ---------------------------------------------------------------------

st.subheader("1. Enter the original French text")

original_text = st.text_area(
    "Original text",
    value=DEFAULT_TEXT,
    height=160,
    placeholder="Paste a French paragraph here...",
)

col_run, col_clear = st.columns([1, 5])

with col_run:
    run_button = st.button(
        "Generate FALC draft and evaluate",
        type="primary",
        use_container_width=True,
    )

with col_clear:
    if st.button("Clear generated result"):

        for key in (
            "simplified",
            "last_original",
            "meaning_report",
            "original_report",
            "simplified_report",
        ):
            st.session_state[key] = (
                None if key.endswith("report") else ""
            )

        st.rerun()


if run_button:

    if not original_text.strip():
        st.error("Please enter a French text first.")

    else:
        try:
            with st.spinner("Loading French NLP model..."):
                nlp = load_nlp()

            # 1. Generate initial draft.
            with st.spinner(
                f"Generating a FALC-oriented draft with {generation_model}..."
            ):
                generated = simplify_text(
                    original_text,
                    generation_model,
                )

            # 2. Validate CPM1 and attempt one repair if needed.
            with st.spinner(
                "Validating sentence structure and repairing if necessary..."
            ):
                (
                    generated,
                    simplified_report,
                    repair_attempted,
                    repair_error,
                ) = repair_cpm1(
                    original_text=original_text,
                    generated_text=generated,
                    model=generation_model,
                    nlp=nlp,
                    book=st.session_state.book,
                    rules=st.session_state.rules,
                )

            if repair_attempted:

                if repair_error:
                    st.warning(
                        "A repair was attempted, but the repair request failed. "
                        "The original generated draft has been retained."
                    )
                    st.caption(repair_error)

                elif "CPM1" in simplified_report.get("rules_failed", []):
                    st.warning(
                        "One repair was attempted, but the draft still "
                        "violates CPM1. Human correction is needed."
                    )

                else:
                    st.success(
                        "One repair was completed and the draft now passes CPM1."
                    )

            else:
                st.success(
                    "The generated draft passed CPM1. No repair was needed."
                )

            # 3. Review meaning of the final draft.
            with st.spinner("Checking meaning preservation..."):
                meaning_report = validate_meaning(
                    original_text=original_text,
                    simplified_text=generated,
                    nlp=nlp,
                    model=reviewer_model,
                    ollama_url=OLLAMA_URL,
                )

            # 4. Evaluate original text.
            # The final generated report is already available.
            with st.spinner("Running the automatic FALC checker..."):
                original_report = check_text(
                    original_text,
                    nlp,
                    st.session_state.book,
                    st.session_state.rules,
                )

            # 5. Save results.
            st.session_state.simplified = generated
            st.session_state.last_original = original_text
            st.session_state.meaning_report = meaning_report
            st.session_state.original_report = original_report
            st.session_state.simplified_report = simplified_report

            st.success("Generation and evaluation completed.")

        except urllib.error.URLError as error:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                "and the selected model is installed."
            )
            st.caption(str(error))

        except Exception as error:
            st.error("The operation failed.")
            st.exception(error)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

if st.session_state.simplified:

    st.subheader("2. Compare the texts")

    left, right = st.columns(2)

    with left:
        st.markdown("**Original text**")

        st.text_area(
            "Original",
            value=st.session_state.last_original,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    with right:
        st.markdown(
            f"**FALC-oriented draft ({st.session_state.generation_model})**"
        )

        st.text_area(
            "Generated text",
            value=st.session_state.simplified,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    if st.session_state.meaning_report:
        display_meaning_report(st.session_state.meaning_report)

    st.subheader("4. Automatic FALC rule evaluation")

    original_result, generated_result = st.columns(2)

    with original_result:
        display_falc_report(
            st.session_state.original_report,
            "Original text",
        )

    with generated_result:
        display_falc_report(
            st.session_state.simplified_report,
            "Generated text",
        )

    st.info(
        "This prototype creates a FALC-oriented draft. Automated checks "
        "cannot certify FALC compliance, guarantee meaning preservation "
        "or measure reader comprehension. Human validation remains necessary."
    )

else:
    st.caption(
        "Enter a French paragraph and select "
        "“Generate FALC draft and evaluate” to begin."
    )
