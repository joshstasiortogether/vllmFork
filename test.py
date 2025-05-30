import sys
import os

# Add the root of your local vllm repo to the path
sys.path.insert(0, os.path.expanduser("~/vllm"))

# Now import your local version
from vllm import LLM
import vllm
from vllm.config import CacheConfig

# Confirm where it's loading from
print("Using vllm from:", vllm.__file__)

# Create a custom cache config
cache_config = CacheConfig(
    block_size=16,  # This is the block size for KV cache management
    gpu_memory_utilization=0.9,
    cache_dtype="auto"
)

# Run a test with the custom config
llm = LLM(
    model="facebook/opt-125m",
    cache_config=cache_config,
    # You might need to adjust these parameters based on your needs
    max_model_len=2048,
)

# Test the model
outputs = llm.generate(["What is the capital of France?"])
print("Generated:", outputs[0].outputs[0].text.strip())

# Optional: Test with a longer sequence to see quantization effects
long_prompt = "Write a detailed explanation of how transformers work in natural language processing. " * 5
outputs = llm.generate([long_prompt])
print("\nLong sequence test:")
print("Generated:", outputs[0].outputs[0].text.strip()) 