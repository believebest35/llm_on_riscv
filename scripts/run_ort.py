import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run ONNX Runtime inference")
parser.add_argument("--base-path", type=str, default="../Qwen3-0.6B-ONNX", help="Base path to model directory")
parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
parser.add_argument("--provider", type=str, default="CPUExecutionProvider", 
                    help="ONNX Runtime execution provider")
args = parser.parse_args()

model_path = Path(args.base_path) / "onnx" / "model.onnx"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

logger.info("Loading tokenizer from %s...", args.base_path)
tokenizer = AutoTokenizer.from_pretrained(args.base_path)

logger.info("Loading ONNX model from %s...", model_path)
session = ort.InferenceSession(str(model_path), providers=[args.provider])

logger.info("\nModel inputs:")
for input_meta in session.get_inputs():
    logger.info("  %s: %s, %s", input_meta.name, input_meta.shape, input_meta.type)

logger.info("\nModel outputs:")
for output_meta in session.get_outputs():
    logger.info("  %s: %s, %s", output_meta.name, output_meta.shape, output_meta.type)

num_layers: int = 28
num_heads: int = 8
head_dim: int = 128

def infer_model_config(session: ort.InferenceSession) -> Tuple[int, int, int]:
    """Infer model configuration from ONNX model metadata."""
    try:
        config = session.get_modelmeta().metadata_props
        if config:
            if hasattr(config, 'num_layers'):
                num_layers = int(config.num_layers)
            if hasattr(config, 'num_heads'):
                num_heads = int(config.num_heads)
            if hasattr(config, 'head_dim'):
                head_dim = int(config.head_dim)
    except Exception as e:
        logger.warning("Could not infer model config from metadata: %s", e)
    return num_layers, num_heads, head_dim

def prepare_inputs(
    input_ids: np.ndarray, 
    past_key_values: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    attention_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Prepare inputs for ONNX model inference."""
    batch_size, seq_length = input_ids.shape
    
    if attention_mask is None:
        attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    
    if past_key_values is None:
        position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
        past_key_values_dict = {}
        for layer_idx in range(num_layers):
            past_key_values_dict[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float32
            )
            past_key_values_dict[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=np.float32
            )
    else:
        past_seq_length = past_key_values[0][0].shape[2]
        position_ids = np.arange(past_seq_length, past_seq_length + seq_length, dtype=np.int64).reshape(1, -1)
        past_key_values_dict = {}
        for layer_idx in range(num_layers):
            past_key_values_dict[f"past_key_values.{layer_idx}.key"] = past_key_values[layer_idx][0]
            past_key_values_dict[f"past_key_values.{layer_idx}.value"] = past_key_values[layer_idx][1]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **past_key_values_dict
    }

def generate_text(prompt: str, max_length: int = 50) -> str:
    """Generate text using ONNX model."""
    logger.info("\nPrompt: %s", prompt)
    
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    
    generated_ids = input_ids.copy()
    past_key_values = None
    
    for step in range(max_length):
        current_attention_mask = np.ones((1, generated_ids.shape[1]), dtype=np.int64)
        onnx_inputs = prepare_inputs(
            np.array([[generated_ids[0, -1]]], dtype=np.int64) if step > 0 else input_ids,
            past_key_values,
            current_attention_mask
        )
        
        start_time = time.time()
        outputs = session.run(None, onnx_inputs)
        inference_time = time.time() - start_time
        
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]
        next_token_id = np.argmax(next_token_logits).item()
        
        generated_ids = np.concatenate([generated_ids, np.array([[next_token_id]])], axis=1)
        
        if next_token_id == tokenizer.eos_token_id:
            logger.info("EOS token generated at step %d", step)
            break
        
        past_key_values = []
        for layer_idx in range(num_layers):
            key = outputs[1 + layer_idx * 2]
            value = outputs[2 + layer_idx * 2]
            past_key_values.append((key, value))
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info("Generated: %s", generated_text)
    
    return generated_text

if __name__ == "__main__":
    logger.info("\n" + "="*50)
    logger.info("Testing ONNX Runtime Inference")
    logger.info("="*50)
    
    test_prompts = [
        "Hello, how are you?",
        "The capital of France is",
        "Once upon a time"
    ]
    
    for prompt in test_prompts:
        generate_text(prompt, max_length=args.max_length)
        logger.info("-" * 50)
