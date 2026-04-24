# Distributed Inference Map

## Entry Points

- TUI chat and agent screens prepare prompts, tokenize, and render streamed tokens.
- `cheetah.orchestration.distributed_inference` owns shard planning, peer model loading, peer RPC generation, and local fallback generation.
- `cheetah.orchestration.model_engine.ModelEngine` owns one shard step: run local layers, serialize hidden state, or sample from final logits.
- `cheetah.orchestration.peer_client.PeerClient` owns TCP commands: `load_model`, `generate_token`, and `clear_model`.
- `cheetah.orchestration.distributed_probe` is the no-TUI test runner.

## Runtime Flow

```text
User prompt
  -> TUI or distributed_probe tokenizes prompt
  -> build_peer_load_plan()
      -> planned_peer_shards()
      -> local shard + remote shard assignments
  -> local load_model_for_backend(..., shard=local_shard)
  -> remote load_model command per peer
      -> PeerClient._handle_load_model_request()
      -> load_model_for_backend(..., shard=remote_shard)
      -> register_generation_runtime()
  -> validate_peer_runtime_fingerprints()
  -> streaming_generate_with_peers()
```

## Per Token Flow

```text
step 0: prefill=true
  local ModelEngine.get_tokens()
    -> run local shard over full prompt
    -> if non-final: return encoded hidden_state + mask + positions
    -> if final: sample token locally

  for each remote peer in order
    -> generate_token RPC
    -> PeerClient decodes tensors for selected backend
    -> remote ModelEngine.get_tokens()
    -> run remote shard
    -> non-final peer returns hidden_state
    -> final peer samples token

step > 0: prefill=false
  local shard receives previous generated token
  local appends attention_mask for that token
  remote final shard samples the next token
  sampled token is appended to input_list and seen_tokens
```

## Shard Convention

- `start_layer` is inclusive.
- `end_layer` is exclusive for transformer layers.
- `total_layers = num_transformer_layers + 1`.
- The extra virtual layer is the final norm/lm_head stage.
- A final transformer shard has `end_layer >= total_layers - 1`.
- Full local model load uses `Shard(model, 0, num_layers, num_layers + 1)`.

Example for 16 transformer layers on two nodes:

```text
local:  start=0  end=8   total=17
remote: start=8  end=16  total=17
final remote applies norm + lm_head because end=16 == total-1
```

## Tensor Transport

- `ModelEngine._encode_tensor()` serializes tensors as base64 + shape + dtype.
- `ModelEngine._decode_tensor()` reconstructs tensors for the requested backend.
- Tinygrad decoded tensors are placed on `TC_TINYGRAD_DEVICE` or CPU.
- Torch bfloat16 payloads are reconstructed from raw bf16 bits through float32, then cast to `torch.bfloat16`.
- Peer runtime fingerprints check config and tokenizer assets before generation.

## Important Invariants

- Generation should use the shard assignments loaded into memory, not silently re-plan around newly discovered peers.
- Prefill resets KV cache on each shard.
- Decode does not reset KV cache.
- Attention mask is one token behind before the local shard processes the previous sampled token; the local shard appends it for decode.
- Sampling happens only on the final shard.
- `seen_tokens` is carried with the request so repetition penalties are applied at the final shard.

## No-TUI Probe

Start a peer node:

```bash
python -m cheetah.orchestration.distributed_probe serve \
  --backend tinygrad \
  --device CPU \
  --bind 0.0.0.0 \
  --port 8765 \
  --peer-id node-b
```

Run generation from the driver node:

```bash
python -m cheetah.orchestration.distributed_probe generate \
  --backend tinygrad \
  --device CPU \
  --port 8766 \
  --peer node-b@192.168.0.5:8765 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain why the sky is blue in one paragraph." \
  --max-new-tokens 64 \
  --temperature 0
```

Torch uses the same script:

```bash
python -m cheetah.orchestration.distributed_probe serve \
  --backend torch \
  --device cpu \
  --port 8765 \
  --peer-id node-b

python -m cheetah.orchestration.distributed_probe generate \
  --backend torch \
  --device cpu \
  --port 8766 \
  --peer node-b@192.168.0.5:8765 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Write a concise test sentence." \
  --max-new-tokens 32 \
  --temperature 0
```

## Files To Read

- `cheetah/orchestration/distributed_inference.py`
- `cheetah/orchestration/model_engine.py`
- `cheetah/orchestration/peer_client.py`
- `cheetah/orchestration/distributed_probe.py`
- `cheetah/models/llm/tinygrad/model.py`
- `cheetah/models/llm/torch/model.py`
