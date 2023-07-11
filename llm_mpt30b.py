from ctransformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download
import llm
import os

from tqdm import tqdm
from functools import partialmethod

DEFAULT_SYSTEM_PROMPT = "A conversation between a user and an LLM-based AI assistant."


@llm.hookimpl
def register_models(register):
    register(Mpt30b(), aliases=("mpt",))


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="mpt30b")
    def mpt30b_():
        "Commands for working with MPT-30B"

    @mpt30b_.command()
    def download():
        "Download the 19GB MPT-30B model file"
        hf_hub_download(
            repo_id="TheBloke/mpt-30B-chat-GGML",
            filename="mpt-30b-chat.ggmlv0.q4_1.bin",
        )


class Mpt30b(llm.Model):
    model_id = "mpt30b"

    class Options(llm.Options):
        verbose: bool = False

    def build_prompt(self, prompt, conversation):
        prompt_lines = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    prompt_lines.append(
                        f"<|im_start|>system\n{prev_response.prompt.system}<|im_end|>\n",
                    )
                    current_system = prev_response.prompt.system
                prompt_lines.append(
                    f"<|im_start|>user\n{prev_response.prompt.prompt}<|im_end|>\n",
                )
                prompt_lines.append(
                    f"<|im_start|>assistant\n{prev_response.text()}<|im_end|>\n",
                )

        system_prompt_to_add = None
        if prompt.system and prompt.system != current_system:
            # User provided a system prompt, use it
            system_prompt_to_add = prompt.system

        # If no system prompt at all, use the default one
        if not current_system and not system_prompt_to_add:
            system_prompt_to_add = DEFAULT_SYSTEM_PROMPT

        if system_prompt_to_add:
            prompt_lines.append(
                f"<|im_start|>system\n{system_prompt_to_add}<|im_end|>\n",
            )
        prompt_lines.extend(
            [
                f"<|im_start|>user\n{prompt.prompt}<|im_end|>\n",
                f"<|im_start|>assistant\n",
            ]
        )
        return prompt_lines

    def execute(self, prompt, stream, response, conversation):
        original_init = tqdm.__init__
        if not prompt.options.verbose:
            # Disable all tqdm output, using this workaround:
            # https://stackoverflow.com/a/67238486/6083
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/mpt-30B-chat-GGML",
                model_type="mpt",
                model_file="mpt-30b-chat.ggmlv0.q4_1.bin",
                config=AutoConfig.from_pretrained(
                    "mosaicml/mpt-30b-chat", context_length=8192
                ),
                local_files_only=True,
            )

            prompt_lines = self.build_prompt(prompt, conversation)

            response._prompt_json = {"prompt_lines": prompt_lines}

            generator = llm_model(
                "".join(prompt_lines),
                temperature=0.2,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                max_new_tokens=512,  # adjust as needed
                seed=42,
                reset=False,  # reset history (cache)
                stream=True,  # streaming per word/token
                threads=int(os.cpu_count() / 2),  # adjust for your CPU
                stop=["<|im_end|>", "|<"],
            )
            for word in generator:
                yield word
        except FileNotFoundError:
            raise llm.ModelError(
                "MPT model not installed - try running 'llm mpt30b download'"
            )
        finally:
            tqdm.__init__ = original_init
