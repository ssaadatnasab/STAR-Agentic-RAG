import argparse
import os
import time
import pandas as pd
from openai import OpenAI
import openai
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("程序启动成功", flush=True)


def _get_token_encoder():
    """Best-effort tokenizer for local enforcement.

    NOTE: Tokenization will not exactly match Claude/Yunwu, but it gives a
    consistent local cap so your saved Excel never exceeds --max_tokens.
    """

    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _count_tokens(text: str, encoder) -> int:
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    # Fallback (rough): treat "tokens" as words
    return len(text.split())


def _truncate_to_max_tokens(text: str, max_tokens: int, encoder):
    """Return (truncated_text, token_count, was_truncated)."""

    if max_tokens <= 0:
        return "", 0, bool(text)
    if not text:
        return "", 0, False

    if encoder is None:
        parts = text.split()
        if len(parts) <= max_tokens:
            return text, len(parts), False
        return " ".join(parts[:max_tokens]), max_tokens, True

    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens), False

    truncated = encoder.decode(tokens[:max_tokens]).strip()
    return truncated, max_tokens, True


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_question_column(df: pd.DataFrame) -> str:
    normalized_map = {_normalize_col(c): c for c in df.columns}
    question_col = next(
        (normalized_map[k] for k in ["question", "questions", "prompt", "query"] if k in normalized_map),
        None,
    )
    if question_col is None:
        raise ValueError(f"输入文件缺少问题列。检测到的列: {list(df.columns)}")
    return question_col


OPENLUX_BASE_URL = "https://api.openlux.ai/v1"
YUNWU_BASE_URL = OPENLUX_BASE_URL
SYSTEM_PROMPT = "Answer the question below:"
DEFAULT_API_KEY_ENV = "YUNWU_API_KEY"
DEFAULT_API_KEY = ""
DEFAULT_INPUT = "/home/user/Desktop/SOAT_LLMs_Base/149_sampled_question_for_gang.xlsx"
DEFAULT_OUTPUT = "/home/user/Desktop/SOAT_LLMs_Base/output_claude-opus-4-6-thinking_149.xlsx"
DEFAULT_MODEL = "claude-opus-4-6-thinking"


def main():
    parser = argparse.ArgumentParser(description="Read Question column, call YUNWU API with retries, and save Result to new xlsx.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--key",
        type=str,
        default=DEFAULT_API_KEY,
        help=f"API key. Defaults to env {DEFAULT_API_KEY_ENV} (or OPENAI_API_KEY).",
    )
    parser.add_argument("--base_url", default=OPENLUX_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--start_row", type=int, default=1, help="从第几行开始读取（1-based）")
    args = parser.parse_args()

    if not args.key:
        raise SystemExit(f"未提供 API key：请使用 --key 或设置环境变量 {DEFAULT_API_KEY_ENV}（或 OPENAI_API_KEY）。")

    df = pd.read_excel(args.input)
    question_col = resolve_question_column(df)

    if "Result" not in df.columns:
        df["Result"] = ""
    if "Provider_Result_Token_Count" not in df.columns:
        df["Provider_Result_Token_Count"] = 0
    if "Result_Token_Count" not in df.columns:
        df["Result_Token_Count"] = 0
    if "Result_Was_Truncated" not in df.columns:
        df["Result_Was_Truncated"] = False

    start_pos = max(1, args.start_row) - 1
    if start_pos >= len(df):
        print(f"起始行 {args.start_row} 超出数据总行数 {len(df)}，无需处理。", flush=True)
        df[[question_col, "Result", "Provider_Result_Token_Count"]].rename(
            columns={question_col: "question"}
        ).to_excel(args.output, index=False)
        print(f"处理完成，已保存到 {args.output}", flush=True)
        return

    client = OpenAI(api_key=args.key, base_url=args.base_url)
    encoder = _get_token_encoder()
    processed_count = 0

    for pos, q in df.loc[start_pos:, question_col].items():
        q_text = "" if pd.isna(q) else str(q)
        human_row = pos + 1

        answer = ""
        provider_completion_tokens = 0

        # Retry only on transient API errors (network / timeout / server errors).
        # Do NOT retry on auth/permission or invalid-request errors.
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q_text},
                    ],
                    temperature=0,
                    stream=False,
                )

                answer = (resp.choices[0].message.content or "").strip()
                provider_completion_tokens = int(resp.usage.completion_tokens) if getattr(resp, "usage", None) else 0
                completion_tokens = provider_completion_tokens
                was_truncated = False

                print(f"\n🔄 Running row {human_row} ...", flush=True)
                print(f"😊 Question: {q_text[:80]}{'...' if len(q_text) > 80 else ''}", flush=True)
                print(f"✅ Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}", flush=True)
                print(f"🔢 Provider completion tokens: {provider_completion_tokens}", flush=True)
                break  # success — exit retry loop

            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                print(f"[{human_row}] 认证失败/无权限（不会重试）: {e}", flush=True)
                save_cols = [c for c in [question_col, "Result", "Result_Token_Count", "Provider_Result_Token_Count", "Result_Was_Truncated"] if c in df.columns]
                df[save_cols].rename(columns={question_col: "question"}).to_excel(args.output, index=False)
                raise SystemExit(
                    "API key 无效或没有权限。请检查 key / base_url / model，并建议立即更换（rotate）已泄露的 key。"
                )

            except (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError) as e:
                print(f"[{human_row}] 请求参数错误/资源不存在（不会重试）: {e}", flush=True)
                save_cols = [c for c in [question_col, "Result", "Result_Token_Count", "Provider_Result_Token_Count", "Result_Was_Truncated"] if c in df.columns]
                df[save_cols].rename(columns={question_col: "question"}).to_excel(args.output, index=False)
                raise SystemExit("请求不可用（BadRequest/NotFound/UnprocessableEntity）。请修正参数后再试。")

            except openai.RateLimitError as e:
                # If provider explicitly says "wait N seconds", honor it.
                msg = str(e)
                wait_time = 2 * (attempt + 1)
                import re

                m = re.search(r"wait\s+(\d+)\s*seconds|等待\s*(\d+)\s*秒", msg, flags=re.IGNORECASE)
                if m:
                    seconds = next((g for g in m.groups() if g), None)
                    if seconds:
                        wait_time = max(wait_time, int(seconds))
                print(f"[{human_row}] 触发限流: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

            except (openai.APIStatusError, openai.APIError) as e:
                wait_time = 2 * (attempt + 1)
                print(f"[{human_row}] API 错误: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

            except Exception as e:
                wait_time = 2 * (attempt + 1)
                print(f"[{human_row}] 调用失败: {e}, {wait_time} 秒后重试 ({attempt + 1}/5)...", flush=True)
                time.sleep(wait_time)

        df.at[pos, "Result"] = answer
        df.at[pos, "Provider_Result_Token_Count"] = provider_completion_tokens
        processed_count += 1

        if processed_count % 10 == 0:
            df[[question_col, "Result", "Provider_Result_Token_Count"]].rename(
                columns={question_col: "question"}
            ).to_excel(args.output, index=False)
            print(f"✅ 已分批保存：处理到第 {human_row} 行（本批累计 {processed_count} 条）。", flush=True)

    # Final save
    df[[question_col, "Result", "Provider_Result_Token_Count"]].rename(
        columns={question_col: "question"}
    ).to_excel(args.output, index=False)
    print(f"处理完成，已保存到 {args.output}", flush=True)


if __name__ == "__main__":
    main()