from click.testing import CliRunner
from llm.cli import cli
from llm_mpt30b import DEFAULT_SYSTEM_PROMPT
import llm
import json
import pytest

DEFAULT_SYSTEM_PROMPT_LINE = f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"


def test_plugin_is_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins"])
    assert result.exit_code == 0, result.output
    names = [plugin["name"] for plugin in json.loads(result.output)]
    assert "llm-mpt30b" in names


@pytest.mark.parametrize(
    "prompt,system,expected",
    (
        (
            "prompt text",
            None,
            [
                # Should use the default system prompt
                DEFAULT_SYSTEM_PROMPT_LINE,
                "<|im_start|>user\nprompt text<|im_end|>\n",
                "<|im_start|>assistant\n",
            ],
        ),
        (
            "prompt text",
            "system prompt",
            [
                "<|im_start|>system\nsystem prompt<|im_end|>\n",
                "<|im_start|>user\nprompt text<|im_end|>\n",
                "<|im_start|>assistant\n",
            ],
        ),
    ),
)
def test_build_prompt_simple(prompt, system, expected):
    model = llm.get_model("mpt")
    lines = model.build_prompt(llm.Prompt(prompt, model, system), None)
    assert lines == expected


def test_build_prompt_conversation():
    model = llm.get_model("mpt")
    conversation = model.conversation()
    conversation.responses = [
        llm.Response.fake(model, "prompt 1", DEFAULT_SYSTEM_PROMPT, "response 1"),
        llm.Response.fake(model, "prompt 2", None, "response 2"),
        llm.Response.fake(model, "prompt 3", None, "response 3"),
    ]
    lines = model.build_prompt(llm.Prompt("prompt 4", model), conversation)
    assert lines == [
        "<|im_start|>system\nA conversation between a user and an LLM-based AI assistant.<|im_end|>\n",
        "<|im_start|>user\nprompt 1<|im_end|>\n",
        "<|im_start|>assistant\nresponse 1<|im_end|>\n",
        "<|im_start|>user\nprompt 2<|im_end|>\n",
        "<|im_start|>assistant\nresponse 2<|im_end|>\n",
        "<|im_start|>user\nprompt 3<|im_end|>\n",
        "<|im_start|>assistant\nresponse 3<|im_end|>\n",
        "<|im_start|>user\nprompt 4<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
