# test_tensor_client.py
import socket
import struct
import json
import torch
import numpy as np
from transformers import AutoTokenizer
import torchtune.generation as ttg

# === Model Setup ===
model_id = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def send_infer_request(sock, input_ids, attention_mask, config_path, layer_start, layer_end):
    input_np = input_ids.cpu().numpy().astype(np.int64)
    mask_np = attention_mask.cpu().numpy().astype(np.int64)
    payload = input_np.tobytes() + mask_np.tobytes()

    header = {
        "command": "infer",
        "model": model_id,
        "config_path": config_path,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "dtype": "int64",
        "shape_input": list(input_np.shape),
        "shape_mask": list(mask_np.shape)
    }

    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(payload)

def recv_tensor(sock):
    raw_len = sock.recv(4)
    if not raw_len:
        raise ConnectionError("Did not receive header length")
    header_len = struct.unpack("!I", raw_len)[0]

    header_data = sock.recv(header_len)
    header = json.loads(header_data.decode("utf-8"))

    shape = header["shape"]
    dtype = header["dtype"]

    if dtype == "bfloat16":
        np_dtype = np.uint16  # raw 16-bit storage
    elif dtype in ("float32", "float"):
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    expected_bytes = np.prod(shape) * np.dtype(np_dtype).itemsize
    data = b""
    while len(data) < expected_bytes:
        chunk = sock.recv(expected_bytes - len(data))
        if not chunk:
            raise ConnectionError("Incomplete tensor data received")
        data += chunk

    flat = np.frombuffer(data, dtype=np_dtype)
    return flat

if __name__ == "__main__":
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.connect("/tmp/tensor_socket")

  # Prompt setup
  prompt = "Hello, how are you?"
  encoded = tokenizer(prompt, return_tensors="pt", padding=True)
  input_ids = encoded["input_ids"]
  attention_mask = encoded["attention_mask"]

  print("Sending input_ids and attention_mask")
  print("input_ids.shape:", input_ids.shape)
  print("attention_mask.shape:", attention_mask.shape)

  config_path = tokenizer.init_kwargs.get("config", tokenizer.name_or_path)

  send_infer_request(
    sock,
    input_ids,
    attention_mask,
    config_path=config_path,
    layer_start=0,
    layer_end=31
  )

  logits = recv_tensor(sock)
  print("Received logits:", logits.shape)

  # === Sampling and decoding ===
  q = torch.empty((logits.size(0), tokenizer.vocab_size), device=logits.device).exponential_(1)
  sampled_tokens = ttg.sample(logits.clone(), temperature=0.8, top_k=50, q=q)
  decoded = tokenizer.decode(sampled_tokens.tolist(), skip_special_tokens=True)

  print("Sampled text:", decoded)
  sock.close()
