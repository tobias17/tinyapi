from pathlib import Path
from typing import List
import json, argparse, random, time, os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16, ModelConfig, TokenSampler
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm

class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".gguf"):
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

MODEL_CFG = ModelConfig(dim=4096, hidden_dim=14336, n_layers=32, n_heads=32, n_kv_heads=8, norm_eps=1e-5, vocab_size=128256, rope_theta=500000.0, max_context=8192)
MODEL_FILES = 1

def build_transformer(model_path:Path, cfg:ModelConfig, files:int, device=None):
  # build model
  model = Transformer(cfg)

  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
    else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(files)], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    assert cfg.n_kv_heads is not None
    weights = convert_from_huggingface(weights, model, cfg.n_heads, cfg.n_kv_heads)
  elif "token_embd.weight" in weights:
    weights = convert_from_gguf(weights, model)
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if '.attention.' in k: v.shard_(device, axis=-1)
        elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=0)
        else: v.shard_(device, axis=None)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
  return model

SAMPLER = TokenSampler(
   temperature=0.95,
   top_k=0,
   top_p=0.0,
   alpha_f=0.0,
   alpha_p=0.0,
)

if __name__ == "__main__":
  Tensor.no_grad = True

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, help="Model path")
  parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
  args = parser.parse_args()

  fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-8b-sfr")
  fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors", "model-00001-of-00004.safetensors", subdir="llama3-8b-sfr")
  fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors", "model-00002-of-00004.safetensors", subdir="llama3-8b-sfr")
  fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors", "model-00003-of-00004.safetensors", subdir="llama3-8b-sfr")
  fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors", "model-00004-of-00004.safetensors", subdir="llama3-8b-sfr")
  model_path = fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json", "model.safetensors.index.json", subdir="llama3-8b-sfr")

  Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")

  tokenizer = Tokenizer(str((model_path if model_path.is_dir() else model_path.parent) / "tokenizer.model"))
  def encode_role(role: str):
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT
  model = build_transformer(model_path, MODEL_CFG, MODEL_FILES, device=device)

  prompt = [tokenizer.bos_id] + encode_message("system", "You are an helpful assistant.")

  user_prompt = "what is the distance between the earth and the moon?"
  print(f"Q: {user_prompt}")
  toks = prompt + encode_message("user", user_prompt) + encode_role("assistant")

  # start_pos = prefill(model, toks[:-1])
  # last_tok = toks[-1]

  while True:
    tok = model(toks, device, SAMPLER)
    toks.append(tok)
    if tok in tokenizer.stop_tokens: break
    print(tokenizer.decode([tok]), end="", flush=True)
  print(flush=True)
