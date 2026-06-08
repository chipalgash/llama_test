"""Shared QLoRA defaults for the Llama 3 thesis experiment."""

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1024

TARGET_MODULES = ["q_proj", "v_proj"]
