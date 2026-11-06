"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
CHAT_SYSTEM_INSTRUCTION = (
    "会話履歴にある情報を踏まえて自然に回答してください。"
    "人物ごとの情報は、質問に答えるため必要な場合だけ使ってください。"
    "記憶していることを毎回列挙したり、「覚えています」と報告したりせず、"
    "ユーザーから確認を求められた場合にだけ説明してください。"
)


def generate(
    client: genai.Client,
    contents: str | list[types.Content],
    *,
    system_instruction: str | None = None,
) -> types.GenerateContentResponse:
    """履歴または質問をAPIへ送り、生成結果を返す。"""
    if system_instruction:
        return client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
    return client.models.generate_content(model=MODEL, contents=contents)


def ask(question: str) -> str:
    """質問を1回だけ送信し、回答を返す。"""
    client = genai.Client(api_key=get_api_key())
    response = generate(client, question)
    return response.text or ""


def chat() -> None:
    """会話履歴を保ちながら、入力された質問に続けて回答する。"""
    client = genai.Client(api_key=get_api_key())
    history: list[types.Content] = []

    # 終了入力や端末からのEOF・割り込みまで、質問を繰り返し受け付ける。
    try:
        while True:
            try:
                question = input("質問> ").strip()
            except EOFError:
                break

            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            history.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text=question)]
                )
            )
            try:
                response = generate(
                    client, history, system_instruction=CHAT_SYSTEM_INSTRUCTION
                )
            except errors.APIError as e:
                # 失敗した質問は履歴に残さず、次の入力へ進む。
                history.pop()
                print(api_error_message(e.code, e.message))
                continue

            answer = response.text or ""
            print(answer)
            usage = response.usage_metadata
            if usage:
                print(
                    f"[トークン] 入力: {usage.prompt_token_count} / "
                    f"出力: {usage.candidates_token_count}"
                )
            history.append(
                types.Content(
                    role="model", parts=[types.Part.from_text(text=answer)]
                )
            )
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話モードで起動する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat()
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
