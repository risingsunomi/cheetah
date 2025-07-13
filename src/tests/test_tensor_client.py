# client.py
import socket
import struct
import json
import torch
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer for LLaMA 3.2 1B
model_id = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def send_infer_request(sock, input_ids, attention_mask, model_name, layer_start, layer_end):
    # Flatten and prepare metadata
    input_np = input_ids.cpu().numpy()
    mask_np = attention_mask.cpu().numpy()

    payload = input_np.tobytes() + mask_np.tobytes()

    header = {
        "command": "infer",
        "model": model_name,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "dtype": str(input_np.dtype),
        "shape_input": list(input_np.shape),
        "shape_mask": list(mask_np.shape)
    }
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(payload)

def recv_tensor(sock):
    raw_len = sock.recv(4)
    header_len = struct.unpack("!I", raw_len)[0]
    header = json.loads(sock.recv(header_len).decode("utf-8"))
    shape = header["shape"]
    dtype = header["dtype"]
    num_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    data = b""
    while len(data) < num_bytes:
        data += sock.recv(num_bytes - len(data))
    return torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())

if __name__ == "__main__":
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/tensor_socket")

    # Prepare sample input
    prompt = "Hello, how are you?"
    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print("Sending input_ids and mask together")
    print("input_ids:", input_ids)
    print("attention_mask:", attention_mask)
    
    send_infer_request(
        sock,
        input_ids,
        attention_mask,
        model_name=model_id,
        layer_start=0,
        layer_end=31
    )

    response = recv_tensor(sock)
    print("Received:", response)
    sock.close()
