"""
Converts encyclopedic_test_subset.json into the two reference JSONL files
required by evaluation_infoseek.py:
  - reference.jsonl      (data_id, answer_eval, data_split)
  - reference_qtype.jsonl (data_id, question_type)
"""

import json
import ujson
import argparse

def prepare_references(input_path: str, out_ref: str, out_qtype: str):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Handle both a list and a dict-of-entries
    if isinstance(data, dict):
        samples = list(data.values())
    else:
        samples = data

    with open(out_ref, 'w') as f_ref, open(out_qtype, 'w') as f_qt:
        for sample in samples:
            data_id = sample['unique_id']

            # answer_eval must be a list of strings
            answer = sample['answer']
            answer_eval = answer if isinstance(answer, list) else [answer]

            # data_split: the eval script routes on endswith('unseen_question')
            # or 'unseen_entity'. We use 'val_unseen_question' as default
            # since this is a generic test subset (adjust if tutor clarifies).
            data_split = "val_unseen_question"

            ref_entry = {
                "data_id": data_id,
                "answer_eval": answer_eval,
                "data_split": data_split,
            }

            # question_type: map Encyclopedic-VQA types to InfoSeek types
            # InfoSeek expects: 'string', 'time', 'numerical'
            qtype_raw = sample.get('question_type', 'automatic').lower()
            if qtype_raw in ('time', 'date'):
                qtype = 'time'
            elif qtype_raw in ('numerical', 'number'):
                qtype = 'numerical'
            else:
                # 'automatic', 'factoid', etc. → treat as string/entity
                qtype = 'string'

            qtype_entry = {
                "data_id": data_id,
                "question_type": qtype,
            }

            f_ref.write(ujson.dumps(ref_entry) + '\n')
            f_qt.write(ujson.dumps(qtype_entry) + '\n')

    print(f"Written {len(samples)} entries to:")
    print(f"  {out_ref}")
    print(f"  {out_qtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                        help="Path to encyclopedic_test_subset.json")
    parser.add_argument('--out_ref', required=True,
                        help="Output path for reference.jsonl")
    parser.add_argument('--out_qtype', required=True,
                        help="Output path for reference_qtype.jsonl")
    args = parser.parse_args()
    prepare_references(args.input_path, args.out_ref, args.out_qtype)