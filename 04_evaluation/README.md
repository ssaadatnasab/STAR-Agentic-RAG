# Stage 4: Answer Evaluation

[← Repository overview](../README.md)

[llm_judger.py](llm_judger.py) compares generated answers with reference answers using an API-based LLM judge and writes evaluation results to Excel.

## Inputs and setup

Install the root package. Supply a workbook containing the question, reference `Answer`, and generated `Result` columns from a baseline or STAR run. Configure a compatible endpoint and judge model. The script reads `YUNWU_API_KEY` or `OPENAI_API_KEY`; credentials can also be supplied with `--api_key`.

## Usage

Run from the repository root:

```bash
python 04_evaluation/llm_judger.py \
  --input_file results/answers.xlsx \
  --output_file results/answers-judged.xlsx \
  --base_url "$OPENAI_BASE_URL" \
  --model YOUR_JUDGE_MODEL
```

Set `OPENAI_BASE_URL` and the credential environment variable before running. Use `--start_index` to choose the starting index, `--save_every` to control checkpoint frequency, and `--system_prompt_file` to supply a judge prompt. Check `--help` for defaults.

## Output and comparison

The output workbook includes the original records and judge results. Keep the judge model, prompt, reference answers, and settings consistent when comparing [baselines](../02_baselines/README.md) with [STAR](../03_star_agentic_rag/README.md). Automated judgments do not establish that an engineering answer is correct.
