import csv, json, random, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,".")
import spacy, falc_checker as fc

N = int(sys.argv[1]); SEED = 42
book, rules = fc.load_rulebook("falc_rulebook.json")
nlp = spacy.load("fr_core_news_md")

rows = list(csv.reader(open("wivico_dataset_v2.tsv", encoding="utf-8"), delimiter="\t"))[1:]
random.Random(SEED).shuffle(rows)
rows = rows[:N]

def run(text):
    rep = fc.check_text(text, nlp, book, rules)
    return rep["score_percent"], set(rep["rules_failed"])

active = list(fc.SENTENCE_RULES) + ["MP8"]
res = {"wiki": [], "viki": [], "viki_simple": []}
t0 = time.time()
for i, r in enumerate(rows):
    wiki, viki, _, viki_label, _ = r[0], r[1], r[2], r[3], r[4]
    sw, fw = run(wiki); sv, fv = run(viki)
    res["wiki"].append((sw, fw)); res["viki"].append((sv, fv))
    if viki_label == "1":
        res["viki_simple"].append((sv, fv))
    if i and i % 250 == 0: print(i, round(time.time()-t0), "s", flush=True)

high = [r for r in active if rules[r]["priority"] == "+++"]
out = {"n_pairs": len(rows), "sample_seed": SEED, "high_priority_rules": high, "sets": {}}
for name, lst in res.items():
    n = len(lst)
    if not n: continue
    per_rule = {rid: round(100*sum(rid in f for _, f in lst)/n, 1) for rid in active}
    out["sets"][name] = {
        "n_texts": n,
        "mean_score": round(sum(s for s, _ in lst)/n, 1),
        "pct_fail_any_high_priority": round(100*sum(bool(set(high) & f) for _, f in lst)/n, 1),
        "pct_fail_CPM1_or_CPM2": round(100*sum(bool({"CPM1","CPM2"} & f) for _, f in lst)/n, 1),
        "pct_fail_any_rule_excl_MP8": round(100*sum(bool(f - {"MP8"}) for _, f in lst)/n, 1),
        "pct_fully_pass_excl_MP8": round(100*sum(not (f - {"MP8"}) for _, f in lst)/n, 1),
        "pct_fail_any_rule": round(100*sum(bool(f) for _, f in lst)/n, 1),
        "pct_fully_pass": round(100*sum(not f for _, f in lst)/n, 1),
        "fail_rate_per_rule_pct": per_rule,
    }
json.dump(out, open("baseline_wivico.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out["sets"], indent=1)[:3000])
