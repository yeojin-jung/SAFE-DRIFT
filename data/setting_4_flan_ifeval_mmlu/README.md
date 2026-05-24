# Setting 4 Data

This directory is for the FLAN-v2 / IFEval / MMLU experiment.

Prepare the JSONL files with:

```bash
python scripts/data/prepare_setting4_flan_ifeval_mmlu.py \
  --flan-source data/raw/flan_v2_data.jsonl \
  --out data/setting_4_flan_ifeval_mmlu/prepared \
  --overwrite
```

The preparation script writes:

```text
prepared/flan_v2_100k.jsonl
prepared/ifeval_validation_teacher_42.jsonl
prepared/ifeval_validation_teacher_metrics_eval.jsonl
prepared/ifeval_test_prompt_only.jsonl
prepared/mmlu_test_200.jsonl
prepared/mmlu_validation_285.jsonl
prepared/manifest.json
```

`flan_v2_data.jsonl` is intentionally not committed here; place it under
`data/raw/` or pass `--flan-source` with an absolute path.

IFEval scoring uses the local evaluator in `src/evaluation/ifeval.py`. It does
not launch OLMES, but it does need the IFEval checker utilities from an
`oe-eval`/OLMES checkout or install. If they are not importable in the active
environment, set `OE_EVAL_ROOT` to that checkout before running evaluation.
