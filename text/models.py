from tinygrad import Tensor, Device, dtypes, nn, Context
from tinygrad.helpers import fetch, colored, GlobalCounters, Timing, DEBUG
from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters
from examples.llama3 import load

from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16

from typing import Dict, Tuple, Set, List, Optional
from transformers import AutoTokenizer
from dataclasses import dataclass
import sys

@dataclass
class ModelInst:
   params: Dict
   weights_url: str
   weights_subdir: str
   num_weights: int
   index_filename: str = "model.safetensors.index.json"
   chunk_filename: str = "model-{i:05d}-of-{num_weights:05d}.safetensors"

@dataclass
class ModelArch:
   tokenizer: str
   instances: Dict[str,ModelInst]

MODELS: Dict[str,ModelArch] = {
   "Qwen": ModelArch(
      tokenizer="Qwen/QwQ-32B-Preview",
      instances={
         "32B": ModelInst(
            params={"dim":5120, "n_heads":40, "n_kv_heads":8, "n_layers":64, "norm_eps":1e-5, "rope_theta":1000000, "vocab_size":152064, "hidden_dim":27648},
            weights_url="https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main",
            weights_subdir="qwq_32b_preview",
            num_weights=17,
         ),
      }
   )
}

TARGET_DTYPE = dtypes.float16

def split_into_blocks(state_dict:Dict[str,Tensor], n_layers:int, device_mem:Dict[str,int]) -> None:
   blocks: List[List[Tensor]] = []
   seen: Set[str] = set()
   phases = [
      ["tok_embeddings.", "freqs_cis"],
      *[[f"layers.{i}."] for i in range(n_layers)],
      ["norm.", "output."],
   ]

   for phase in phases:
      block = []
      for k, w in state_dict.items():
         for start in phase:
            if k.startswith(start):
               assert k not in seen, f"Duplicated key {k} found in phase {phase}"
               seen.add(k)
               block.append(w)
               break
      assert len(block) > 0, f"Found 0 keys for phase {phase}"
      blocks.append(block)

   missing = set(state_dict.keys()).difference(seen)
   assert len(missing) == 0, f"Missed the following keys: {list(missing)}"

   mem_rem = {k:v for k,v in device_mem.items()}
   dev_key = list(mem_rem.keys())[0]
   for b in blocks:
      sz = sum(w.lazydata.size * TARGET_DTYPE.itemsize for w in b)
      while True:
         if sz <= mem_rem[dev_key]:
            for w in b:
               w.replace(w.to(dev_key).cast(TARGET_DTYPE))
            mem_rem[dev_key] -= sz
            break

         mem_rem.pop(dev_key)
         assert len(mem_rem) > 0, f"Ran out of memory loading model"
         dev_key = list(mem_rem.keys())[0]

def shard_across(state_dict:Dict[str,Tensor], devices:Tuple[str,...]) -> None:
   def get_axis(key) -> Optional[int]:
      if '.attention.' in key: return -1
      if '.feed_forward.w1.' in key: return 0
      if '.feed_forward.w3.' in key: return 0
      if '.feed_forward.' in key: return -1
      if 'tok_embeddings.weight' in key: return 0
      if 'output.weight' in key: return 0
      return None

   for k, w in state_dict.items():
      w.replace(w.shard(devices, axis=get_axis(k)).cast(TARGET_DTYPE))

def load_model(inst:ModelInst, device_mem:Dict[str,int]) -> Transformer:
   index_filename = "model.safetensors.index.json"
   model_path = fetch(f"{inst.weights_url}/{inst.index_filename}?download=true", index_filename, subdir=inst.weights_subdir)
   for i in range(1, inst.num_weights+1):
      filename = inst.chunk_filename.format(i=i, num_weights=inst.num_weights)
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)
   
   model = Transformer(**inst.params, linear=nn.Linear)
   weights = fix_bf16(convert_from_huggingface(load(str(model_path)), model, inst.params["n_heads"], inst.params["n_kv_heads"], permute_layers=False))

   state_dict = get_state_dict(model)
   # split_into_blocks(state_dict, inst.params["n_layers"])
   shard_across(state_dict, tuple(device_mem.keys()))

   load_state_dict(model, weights, strict=False, consume=True)
   return model

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("Performs a test load and generation of the specified model")
   model_options: Dict[str,Tuple[ModelArch,ModelInst]] = {}
   for arch_name, arch_val in MODELS.items():
      for inst_name, inst_val in arch_val.instances.items():
         model_options[f"{arch_name}-{inst_name}"] = (arch_val,inst_val)
   parser.add_argument('model', choices=list(model_options.keys()), help="Which model to load and test")
   parser.add_argument('--gpus', type=int, default=4, help="Number of GPUs allowed to be used")
   parser.add_argument('--dev-offset', type=int, default=0, help="Skips the first N devices to load the weights")
   parser.add_argument('--vram-limit', type=int, default=16, help="Amount of VRAM in GB per GPU to load weights")

   parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
   parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
   parser.add_argument("--prompt", type=str, default="The distance from the earth to the moon ", help="Phrase to start with")
   parser.add_argument("--timing", action="store_true", help="Print timing per token")
   
   args = parser.parse_args()

   # Load model
   with Context(BEAM=0):
      arch, inst = model_options[args.model]
      device_mem: Dict[str,int] = {}
      for i in range(args.dev_offset, args.dev_offset + args.gpus):
         device_mem[f"{Device.DEFAULT}:{i}"] = args.vram_limit*(1024**3)
      devices = tuple(device_mem.keys())
      transformer = load_model(inst, device_mem)

      tokenizer = AutoTokenizer.from_pretrained(arch.tokenizer)
      param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(transformer))

   outputted = args.prompt
   start_pos, toks = 0, tokenizer(outputted)["input_ids"]
   print(outputted, end="", flush=True)

   tok_tensor: Optional[Tensor] = None
   for i in range(args.count):
      GlobalCounters.reset()

      if args.timing: print("")
      st = GlobalCounters.time_sum_s
      next_tok = Tensor([toks[start_pos:]]) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
      with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
         with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "") +
                     f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB" +
                     (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):
            tok_tensor = transformer(next_tok.shard(devices), start_pos, args.temperature)
            tok = tok_tensor.item()

      # use the kv cache
      start_pos = len(toks)

      # add the new token
      toks.append(tok)

      cur = tokenizer.decode(toks, skip_special_tokens=True)
      sys.stdout.write(cur[len(outputted):])
      sys.stdout.flush()
      outputted = cur

   print()
