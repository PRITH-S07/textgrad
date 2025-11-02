import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from textgrad.engine_experimental.base import EngineLM, cached
import diskcache as dc
from typing import Union, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class MistralEngine(EngineLM):
    """
    TextGrad engine for Mistral-7B-Instruct-v0.3 (or any Hugging Face chat model).
    
    Usage:
        engine = MistralEngine(
            model_string="mistralai/Mistral-7B-Instruct-v0.3",
            device="cuda",  # or "cpu"
            load_in_8bit=False,  # Set to True for quantization
            cache=True
        )
        
        response = engine.generate("What is 2+2?")
    """
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "mistralai/Mistral-7B-Instruct-v0.3",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        is_multimodal: bool = False,
        cache: Union[dc.Cache, bool] = False,
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
    ):
        """
        Initialize the Mistral engine.
        
        Args:
            model_string: Hugging Face model ID
            system_prompt: Default system prompt
            device: Device to load model on ("cuda" or "cpu")
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            is_multimodal: Whether the model supports multimodal inputs
            cache: Whether to cache responses
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache
        )
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        print(f"Loading {model_string}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        model_kwargs = {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_string,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"Model loaded successfully on {device}!")

    def mistral_generate(
        self, 
        content: str, 
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the Mistral model.
        
        Args:
            content: User message
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        if temperature is None:
            temperature = self.temperature
            
        if max_tokens is None:
            max_tokens = self.max_new_tokens
        
        # Format messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
        self,
        content: str,
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 32768,
        top_p: float = 0.99,
        **kwargs
    ):
        """Generate from a single text prompt."""
        return self.mistral_generate(
            content,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 32768,
        top_p: float = 0.99,
        **kwargs
    ):
        """
        Generate from multiple inputs (for multimodal support).
        For text-only models, we concatenate the text inputs.
        """
        # Extract text content and concatenate
        text_content = []
        for item in content:
            if isinstance(item, str):
                text_content.append(item)
            elif isinstance(item, bytes):
                # Skip binary content for text-only models
                continue
        
        combined_content = "\n".join(text_content)
        
        return self.mistral_generate(
            combined_content,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

    def __call__(self, content, **kwargs):
        """Make the engine callable."""
        return self.generate(content, **kwargs)


# Example usage function
def example_usage():
    """Example of how to use the MistralEngine with TextGrad."""
    
    # Initialize the engine
    engine = MistralEngine(
        model_string="mistralai/Mistral-7B-Instruct-v0.3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,  # Set to True if you have limited VRAM
        cache=True
    )
    
    # Simple generation
    response = engine.generate("What is the capital of France?")
    print("Response:", response)
    
    # With custom system prompt
    response = engine.generate(
        "Write a haiku about programming",
        system_prompt="You are a creative poet."
    )
    print("\nHaiku:", response)
    
    # Use with TextGrad
    # from textgrad import set_backward_engine
    # set_backward_engine(engine)


if __name__ == "__main__":
    example_usage()