# save as tools/msmarco_to_jsonl.py
import argparse, json, sys

def parse_answers_field(raw: str):
    # Split multiple answers if user used " ||| " as a delimiter; otherwise single answer
    raw = raw.strip()
    if " ||| " in raw:
        parts = [p.strip() for p in raw.split(" ||| ") if p.strip()]
        return parts if parts else [""]
    return [raw] if raw else [""]

def convert_answers_tsv_to_jsonl(in_path: str, out_path: str):
    """
    Input TSV:
        <qid>\t<answer or 'a1 ||| a2 ||| a3'>
    Output JSONL:
        {"query_id": "<qid>", "answers": ["a1", "a2", ...]}
    """
    n, m = 0, 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[warn] skipping line (needs qid<TAB>answers): {line[:120]}...", file=sys.stderr)
                continue
            qid, answers_raw = parts[0], parts[1]
            answers = parse_answers_field(answers_raw)
            obj = {"query_id": str(qid), "answers": answers}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            m += len(answers)
    print(f"[ok] wrote {n} lines to {out_path} (total answers: {m})")

def main():
    ap = argparse.ArgumentParser(description="Convert MS MARCO answers TSV to evaluator JSONL.")
    ap.add_argument("--answers", required=True, help="Path to msmarco answers tsv (qid<TAB>answer[ ||| answer2 ...])")
    ap.add_argument("--out", required=True, help="Path to write JSONL (reference.jsonl)")
    args = ap.parse_args()
    convert_answers_tsv_to_jsonl(args.answers, args.out)

if __name__ == "__main__":
    main()
