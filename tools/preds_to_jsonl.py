# save as tools/preds_to_jsonl.py
import argparse, json

def convert_preds_tsv_to_jsonl(in_path: str, out_path: str):
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid, pred = parts[0], parts[1]
            obj = {"query_id": str(qid), "answers": [pred]}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"[ok] wrote {n} predictions to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert qid<TAB>prediction TSV to evaluator JSONL.")
    ap.add_argument("--preds", required=True, help="Path to predictions tsv (qid<TAB>prediction)")
    ap.add_argument("--out", required=True, help="Path to write JSONL (candidates.jsonl)")
    args = ap.parse_args()
    convert_preds_tsv_to_jsonl(args.preds, args.out)

if __name__ == "__main__":
    main()
