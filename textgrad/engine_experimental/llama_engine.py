import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from textgrad.engine_experimental.base import EngineLM, cached
import diskcache as dc
from typing import Union, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class LlamaEngine(EngineLM):
    """
    TextGrad engine for Llama-3.2-1B-Instruct (or any HF chat-style Llama 3 model).

    Usage:
        from llama_engine import LlamaEngine

        engine = LlamaEngine(
            model_string="meta-llama/Llama-3.2-1B-Instruct",
            device="cuda",          # or "cpu"
            load_in_8bit=False,     # quantization if you want
            cache=True
        )

        response = engine.generate("What is 2+2?")
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        is_multimodal: bool = False,
        cache: Union[dc.Cache, bool] = False,
        max_new_tokens: Optional[int] = None,  # None → no explicit generation limit
        temperature: float = 0.7,
    ):
        """
        Initialize the Llama engine.

        Important: we do NOT explicitly cap input length (no truncation) and
        we do NOT force a max_new_tokens unless you pass one. The only limits
        will be from the underlying model / hardware.
        """
        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading {model_string}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)

        # Set padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left padding often works better for generation with causal LMs
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        # Load model with optional quantization
        model_kwargs = {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = (
                torch.float16 if device == "cuda" else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_string,
            **model_kwargs,
        )

        # Move to device if not using device_map
        if not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(device)

        self.model.eval()
        print(f"Model loaded successfully on {device}!")

    def llama_generate(
        self,
        content: str,
        system_prompt: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """
        Generate a response using the Llama 3 model.

        We:
        - Try to use the HF chat template if available.
        - Fallback to a plain text prompt if chat_template is missing.
        - Do NOT truncate the input.
        - Only set max_new_tokens if you explicitly pass it OR if self.max_new_tokens is set.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        if temperature is None:
            temperature = self.temperature

        # Do not force any global max unless user / init specifies it
        effective_max_new_tokens = (
            max_tokens if max_tokens is not None else self.max_new_tokens
        )

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # ---- Try chat_template, fallback to plain text ----
        prompt = None
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            # chat_template not set or broken → fall back
            prompt = None

        if prompt is None:
            # Simple fallback: concatenate system + user into a single text prompt
            prompt = f"{system_prompt}\n\n{content}"

        # Tokenize WITHOUT truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        gen_kwargs = dict(
            **inputs,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Only set max_new_tokens if we actually have a value
        if effective_max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = effective_max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        # Decode only the generated continuation
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text.strip()

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
        self,
        content: str,
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        top_p: float = 0.99,
        **kwargs,
    ):
        """Generate from a single text prompt."""
        return self.llama_generate(
            content,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        top_p: float = 0.99,
        **kwargs,
    ):
        """
        Generate from multiple inputs (for multimodal support).
        For text-only models, we concatenate the text inputs.
        """
        text_content = []
        for item in content:
            if isinstance(item, str):
                text_content.append(item)
            elif isinstance(item, bytes):
                # ignore raw bytes for text-only model
                continue

        combined_content = "\n".join(text_content)

        return self.llama_generate(
            combined_content,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    def __call__(self, content, **kwargs):
        """Make the engine callable (matches MistralEngine behaviour)."""
        return self.generate(content, **kwargs)


# Example usage
def example_usage():
    engine = LlamaEngine(
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,
        cache=True,
        max_new_tokens=None,  # <- no explicit limit; you can also set a huge number here
    )

    resp = engine.generate("What is the capital of France?")
    print("Response:", resp)

    resp = engine.generate(
        "Write a haiku about programming.",
        system_prompt="You are a creative poet.",
    )
    print("\nHaiku:", resp)


if __name__ == "__main__":
    example_usage()
