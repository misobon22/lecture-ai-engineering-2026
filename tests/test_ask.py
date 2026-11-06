from types import SimpleNamespace

from google.genai import errors

from app import ask as ask_module


def test_chat_keeps_successful_history_and_drops_failed_question(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 500
        message = "temporary failure"

    class FakeClient:
        class Models:
            def generate_content(self, *, model, contents, config=None):
                calls.append((list(contents), config))
                if len(calls) == 1:
                    raise FakeAPIError()
                return SimpleNamespace(
                    text="回答",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=12, candidates_token_count=4
                    ),
                )

        models = Models()

    answers = iter(["失敗する質問", "次の質問", "続けて質問", "quit"])
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module, "api_error_message", lambda code, message: "APIエラー")
    monkeypatch.setattr(errors, "APIError", FakeAPIError)

    ask_module.chat()

    assert [[content.role for content in call] for call, _ in calls] == [
        ["user"],
        ["user"],
        ["user", "model", "user"],
    ]
    assert all(
        config.system_instruction == ask_module.CHAT_SYSTEM_INSTRUCTION
        for _, config in calls
    )
    output = capsys.readouterr().out
    assert "APIエラー" in output
    assert "[トークン] 入力: 12 / 出力: 4" in output