import pandas as pd
import json

df = pd.read_csv("falc_comparison_results.csv")

def has_rule(value, rule_id):
    if pd.isna(value):
        return False
    return rule_id in json.loads(value)

df["standard_mp8"] = df["standard_failed_rules"].apply(
    lambda x: has_rule(x, "MP8")
)

df["falc_mp8"] = df["falc_failed_rules"].apply(
    lambda x: has_rule(x, "MP8")
)

cases = df[df["falc_mp8"] & ~df["standard_mp8"]]

print("FALC fails MP8, standard passes:", len(cases))

for _, row in cases.head(10).iterrows():
    print("\n" + "=" * 60)
    print("STANDARD:")
    print(repr(row["standard_text"]))
    print("FALC:")
    print(repr(row["falc_text"]))