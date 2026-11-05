import argparse
import json
import re
import sys
from pathlib import Path

import spacy
from wordfreq import zipf_frequency

RULEBOOK = Path(__file__).with_name("falc_rulebook.json")

# --------------------------------------------------------------------------- seed lists
# Seed lists
ANGLICISMS = {"meeting", "overbooké", "overbooked", "geek", "deadline", "email", "mail",
              "feedback", "call", "planning", "week-end", "weekend", "coach", "burn-out",
              "burnout", "manager", "briefing", "team", "teaming", "business", "online"}
TEXTO = {"jtm", "mdr", "slt", "bjr", "stp", "svp", "pk", "koi", "cc", "pq", "tkt", "dsl", "bcp", "ptdr"}
ABBREV = re.compile(
    r"(?<![\w])(Dr|Pr|Me|Mme|Mlle|M\.|etc\.|cf\.|c\.-à-d\.|ex\.|env\.|tél\.|St\.|Ste\.|av\.|bd\.)(?![\w])")
NEG_WORDS = {"pas", "plus", "jamais", "rien", "personne", "aucun", "aucune", "guère"}
DATE_NUMERIC = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s?%|pour cent", re.I)
LARGE_NUMBER = re.compile(r"\b\d{1,3}(?:[ \u00a0.]\d{3})+\b|\b\d{4,}\b")
ROMAN = None 


# --------------------------------------------------------------------------- helpers
def params(rule):
    return (rule.get("detector") or {}).get("params") or {}


def finding(rule, evidence, token=None):
    return {"rule_id": rule["id"], "evidence": evidence,
            "char_start": token.idx if token is not None else None}


def predicate_count(sent):
    """Count finite verbal predicates (compound tenses count once)."""
    n = 0
    for t in sent:
        morph = t.morph.get("VerbForm")
        has_aux = any(c.dep_ in ("aux:tense", "aux:pass", "aux") for c in t.children)
        if t.pos_ == "VERB" and ("Fin" in morph or ("Part" in morph and has_aux)):
            n += 1
        elif t.pos_ == "AUX" and t.dep_ in ("cop", "ROOT"):
            n += 1
    return n


# --------------------------------------------------------------------------- sentence rules
# Each function takes (rule, sent) and returns a list of findings.
def cpm1(rule, sent):
    p = params(rule)
    words = [t for t in sent if t.is_alpha]
    preds = predicate_count(sent)
    out = []
    if len(words) > p.get("max_words", 15):
        out.append(finding(rule, f"{len(words)} words (max {p.get('max_words', 15)})"))
    if preds > p.get("max_finite_verbs", 1):
        out.append(finding(rule, f"{preds} verbal predicates (max {p.get('max_finite_verbs', 1)})"))
    return out


def cpm2(rule, sent):
    # word frequency
    p = params(rule)
    thr = p.get("min_zipf_frequency", 3.5)
    out = []
    for t in sent:
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and t.is_alpha:
            f = zipf_frequency(t.lemma_.lower(), "fr")
            if f < thr:
                out.append(finding(rule, f"'{t.text}' rare word (zipf {f:.2f} < {thr})", t))
    return out


def cpm8(rule, sent):
    return [finding(rule, f"'{t.text}' anglicism", t) for t in sent if t.lower_ in ANGLICISMS]


def cpm9(rule, sent):
    out = []
    for t in sent:
        if t.lower_ in TEXTO or re.fullmatch(r"[A-Za-zÀ-ÿ]+\d+[A-Za-zÀ-ÿ0-9]*", t.text):
            out.append(finding(rule, f"'{t.text}' texto-style word", t))
    return out


def cpm12(rule, sent):
    p = params(rule)
    has_ne = any(t.lower_ in ("ne", "n'", "n’") for t in sent)
    negs = [t for t in sent if (t.lower_ in NEG_WORDS and has_ne) or t.lower_ == "sans"]
    if len(negs) > p.get("max_negations_per_sentence", 1):
        return [finding(rule, f"{len(negs)} negations: " + ", ".join(t.text for t in negs), negs[0])]
    return []


def cpm13(rule, sent):
    # spaCy fr marks passives with nsubj:pass, aux:pass and often obl:agent.
    return [finding(rule, f"passive construction around '{t.text}'", t)
            for t in sent if t.dep_ in ("nsubj:pass", "aux:pass", "obl:agent")][:1]


def cpm14(rule, sent):
    chars = params(rule).get("forbidden_chars", [";", "(", ")", "…"])
    return [finding(rule, f"'{c}' used", None) for c in chars if c in sent.text]


def cpm15(rule, sent):
    # The French parser labels list items inconsistently (conj / appos / xcomp / advcl),
    # so we use commas as a robust signal: n items need n-1 separators.
    n_min = params(rule).get("min_items_for_list", 3)
    commas = [t for t in sent if t.text == ","]
    if len(commas) >= n_min - 1:
        return [finding(rule, f"{len(commas)} commas: possible list of {len(commas) + 1} items, use bullets", commas[0])]
    return []


def cpm16(rule, sent):
    return [finding(rule, f"'{t.text}' number in letters", t)
            for t in sent if t.pos_ == "NUM" and t.text.isalpha()]


def cpm17(rule, sent):
    return [finding(rule, f"'{t.text}' acronym", t)
            for t in sent if t.text.isupper() and t.text.isalpha() and 2 <= len(t.text) <= 3
            and not ROMAN.fullmatch(t.text)]


def cpm18(rule, sent):
    return [finding(rule, f"'{m.group(0)}' abbreviation", None) for m in ABBREV.finditer(sent.text)]


def cpm19(rule, sent):
    return [finding(rule, f"'{m.group(0)}' numeric date", None) for m in DATE_NUMERIC.finditer(sent.text)]


def cpm20(rule, sent):
    out = [finding(rule, f"'{m.group(0)}' percentage", None) for m in PERCENT.finditer(sent.text)]
    for m in LARGE_NUMBER.finditer(sent.text):
        raw = re.sub(r"[ \u00a0.]", "", m.group(0))
        if raw.isdigit() and 1900 <= int(raw) <= 2100 and len(raw) == 4:
            continue  # looks like a year
        if raw.isdigit() and int(raw) >= params(rule).get("large_number_min", 1000):
            out.append(finding(rule, f"'{m.group(0)}' large number", None))
    return out


def cpm21(rule, sent):
    chars = ["&", "<", ">", "?", "#"]
    return [finding(rule, f"'{c}' special character", None) for c in chars if c in sent.text]


def mp5(rule, sent):
    # 2-3 letter capitals are handled as acronyms (CPM17). Here: long words in capitals.
    return [finding(rule, f"'{t.text}' word in capitals", t)
            for t in sent if t.text.isupper() and t.text.isalpha() and len(t.text) >= 4]


def mp7(rule, sent):
    return [finding(rule, f"'{t.text}' roman numeral", t)
            for t in sent if t.text.isupper() and t.text.isalpha() and len(t.text) >= 2
            and ROMAN.fullmatch(t.text)]


SENTENCE_RULES = {"CPM1": cpm1, "CPM2": cpm2, "CPM8": cpm8, "CPM9": cpm9, "CPM12": cpm12,
                  "CPM13": cpm13, "CPM14": cpm14, "CPM15": cpm15, "CPM16": cpm16,
                  "CPM17": cpm17, "CPM18": cpm18, "CPM19": cpm19, "CPM20": cpm20,
                  "CPM21": cpm21, "MP5": mp5, "MP7": mp7}


# --------------------------------------------------------------------------- main engine
def load_rulebook(path=RULEBOOK):
    global ROMAN
    book = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = {r["id"]: r for r in book["rules"]}
    ROMAN = re.compile(rules["MP7"]["detector"]["params"]["pattern"])
    return book, rules


def check_text(text, nlp, book, rules):
    lines = [l for l in text.splitlines() if l.strip()]
    sentences, mp8_findings = [], []

    for li, line in enumerate(lines, 1):
        doc = nlp(line)
        sents = [s for s in doc.sents if sum(t.is_alpha for t in s) >= 2]
        if len(sents) > 1:
            mp8_findings.append({"rule_id": "MP8", "evidence": f"line {li} has {len(sents)} sentences",
                                 "char_start": None})
        for s in sents:
            sentences.append((li, s))

    report_sents = []
    failed_rules = {}
    for idx, (li, s) in enumerate(sentences, 1):
        fails = []
        for rid, fn in SENTENCE_RULES.items():
            for f in fn(rules[rid], s):
                fails.append(f)
                failed_rules.setdefault(rid, []).append(f)
        report_sents.append({"sentence_id": f"s{idx}", "line": li, "text": s.text.strip(),
                             "failures": fails})
    if mp8_findings:
        failed_rules["MP8"] = mp8_findings

    # ---- scoring (same logic as the official sheet: 3 / 2 / 1 points)
    active = [rid for rid in list(SENTENCE_RULES) + ["MP8"]]
    total = sum(rules[r]["weight"] for r in active)
    passed = sum(rules[r]["weight"] for r in active if r not in failed_rules)
    by_priority = {}
    for pr in ("+++", "++", "+"):
        ids = [r for r in active if rules[r]["priority"] == pr]
        ok = [r for r in ids if r not in failed_rules]
        by_priority[pr] = {"passed": len(ok), "total": len(ids)}

    return {
        "score_percent": round(100 * passed / total, 1) if total else None,
        "rules_checked": len(active),
        "rules_failed": sorted(failed_rules),
        "by_priority": by_priority,
        "sentences": report_sents,
        "document_level": mp8_findings,
        "note": "Phase 1 rules only (automatic). A passing score does not mean the text is FALC: "
                "phase 2/3 rules and review by people with intellectual disabilities are still required.",
    }


def print_report(rep, rules):
    print(f"\nFALC score (phase 1 rules): {rep['score_percent']} %   "
          f"({rep['rules_checked']} rules checked, {len(rep['rules_failed'])} failed)")
    for pr, v in rep["by_priority"].items():
        print(f"  priority {pr}: {v['passed']}/{v['total']} rules passed")
    print()
    for s in rep["sentences"]:
        status = "OK  " if not s["failures"] else "FAIL"
        print(f"[{status}] {s['sentence_id']}: {s['text']}")
        for f in s["failures"]:
            print(f"        - {f['rule_id']} ({rules[f['rule_id']]['priority']}): {f['evidence']}")
    for f in rep["document_level"]:
        print(f"[FAIL] document: {f['rule_id']} - {f['evidence']}")


def main():
    ap = argparse.ArgumentParser(description="HERMES FALC checker (phase 1 rules)")
    ap.add_argument("text", nargs="?", help="text to check")
    ap.add_argument("--file", help="read text from a file")
    ap.add_argument("--json", help="write the full report to this JSON file")
    ap.add_argument("--model", default="fr_core_news_md")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        sys.exit("Give a text or --file")
    book, rules = load_rulebook()
    nlp = spacy.load(args.model)
    rep = check_text(text, nlp, book, rules)
    print_report(rep, rules)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.json}")


if __name__ == "__main__":
    main()
