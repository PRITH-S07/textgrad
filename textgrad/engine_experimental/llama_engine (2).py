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
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ):
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_string)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

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

        if not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(device)

        self.model.eval()
        print(f"Model loaded successfully on {device}!")

    def get_choice_logprobs(self, prompt: str, choices: List[str] = ["A", "B", "C", "D"]) -> List[float]:
        """
        Get log probabilities for each choice token given the prompt.
        Used for likelihood-based evaluation (like MMLU/ARC benchmarks).
        """
        # Tokenize once
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get the logits for the last position (predicting next token)
        last_logits = logits[0, -1, :]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(last_logits, dim=-1)
        
        logprobs = []
        for choice in choices:
            choice_token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(choice_token_ids) > 0:
                choice_token_id = choice_token_ids[0]
                choice_logprob = log_probs[choice_token_id].item()
            else:
                choice_logprob = float('-inf')
            logprobs.append(choice_logprob)
        
        return logprobs

    def predict_by_likelihood(self, prompt: str, choices: List[str] = ["A", "B", "C", "D"]) -> str:
        """Return the choice with highest likelihood."""
        logprobs = self.get_choice_logprobs(prompt, choices)
        best_idx = logprobs.index(max(logprobs))
        return choices[best_idx]

    def llama_generate(
        self,
        content: str,
        system_prompt: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        if system_prompt is None:
            system_prompt = self.system_prompt

        if temperature is None:
            temperature = self.temperature

        effective_max_new_tokens = (
            max_tokens if max_tokens is not None else self.max_new_tokens
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        prompt = None
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            prompt = None

        if prompt is None:
            prompt = f"{system_prompt}\n\n{content}"

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

        if effective_max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = effective_max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

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
        text_content = []
        for item in content:
            if isinstance(item, str):
                text_content.append(item)
            elif isinstance(item, bytes):
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
        return self.generate(content, **kwargs)