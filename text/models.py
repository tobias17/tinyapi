from tinygrad import Tensor, Device, dtypes, nn, Context
from tinygrad.helpers import fetch, colored, GlobalCounters, Timing, DEBUG, tqdm
from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters
from examples.llama3 import load

from extra.models.llama import Transformer, ModelConfig, convert_from_huggingface, fix_bf16

from typing import Dict, Tuple, Set, List, Optional
from transformers import AutoTokenizer
from dataclasses import dataclass, field

@dataclass
class ModelInst:
   config: ModelConfig
   weights_url: str
   weights_subdir: str
   num_weights: int
   index_filename: str = "model.safetensors.index.json"
   chunk_filename: str = "model-{i:05d}-of-{num_weights:05d}.safetensors"
   extra_filenames: List[str] = field(default_factory=lambda: ["tokenizer.json", "tokenizer_config.json"])


TARGET_DTYPE = dtypes.float16
MODELS: Dict[str,ModelInst] = {
   # "Qwen-32B": ModelInst(
   #    params={"dim":5120, "n_heads":40, "n_kv_heads":8, "n_layers":64, "norm_eps":1e-5, "rope_theta":1000000, "vocab_size":152064, "hidden_dim":27648},
   #    weights_url="https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main",
   #    weights_subdir="qwq_32b_preview",
   #    num_weights=17,
   # ),
   # "Qwen-R1-32B": ModelInst(
   #    params={"dim":5120, "n_heads":40, "n_kv_heads":8, "n_layers":64, "norm_eps":1e-5, "rope_theta":1000000, "vocab_size":152064, "hidden_dim":27648},
   #    weights_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/resolve/main",
   #    weights_subdir="deepseek_r1_qwen_32b",
   #    num_weights=8,
   #    chunk_filename="model-{i:05d}-of-{num_weights:06d}.safetensors", # WHY????
   #    # Prompt template: https://unsloth.ai/blog/deepseekr1-dynamic
   # ),
   "Mistral-24B": ModelInst(
      config=ModelConfig(dim=5120, hidden_dim=32768, n_layers=40, n_heads=32, head_dim=128, n_kv_heads=8, norm_eps=1e-5, vocab_size=131072, rope_theta=100000000.0, max_context=32768),
      weights_url="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/resolve/main",
      weights_subdir="mistral_small_24b_instruct",
      num_weights=10,
   ),
}


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
   def get_axis(key:str) -> Optional[int]:
      if '.attention.' in key: return None
      if '.attention.' in key: return -1
      if '.feed_forward.w1.' in key: return 0
      if '.feed_forward.w3.' in key: return 0
      if '.feed_forward.' in key: return -1
      if 'tok_embeddings.weight' in key: return 0
      if 'output.weight' in key: return 0
      return None

   for k, w in state_dict.items():
      w.replace(w.shard(devices, axis=get_axis(k)).cast(TARGET_DTYPE))


def load_model(inst:ModelInst, device_mem:Dict[str,int], skip_load:bool=False) -> Tuple[Transformer,AutoTokenizer]:
   index_filename = "model.safetensors.index.json"
   model_path = fetch(f"{inst.weights_url}/{inst.index_filename}?download=true", index_filename, subdir=inst.weights_subdir)
   for i in range(1, inst.num_weights+1):
      filename = inst.chunk_filename.format(i=i, num_weights=inst.num_weights)
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)

   for filename in inst.extra_filenames:
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)
   tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent))

   model = Transformer(inst.config)
   # updated_layers = []
   # for layer in model.layers:
   #    head_dim = inst.params["dim"] // inst.params["n_heads"]
   #    layer.attention.wq = nn.Linear(inst.params["dim"], inst.params["n_heads"]    * head_dim, bias=True)
   #    layer.attention.wk = nn.Linear(inst.params["dim"], inst.params["n_kv_heads"] * head_dim, bias=True)
   #    layer.attention.wv = nn.Linear(inst.params["dim"], inst.params["n_kv_heads"] * head_dim, bias=True)
   #    updated_layers.append(layer)
   # model.layers = updated_layers

   # split_into_blocks(get_state_dict(model), inst.params["n_layers"])
   shard_across(get_state_dict(model), tuple(device_mem.keys()))

   weights = fix_bf16(convert_from_huggingface(load(str(model_path)), model, inst.config.n_heads, inst.config.n_kv_heads, permute_layers=False))
   if not skip_load:
      load_state_dict(model, weights, strict=False, consume=True)

   return model, tokenizer

TEMPERATURE = 0.3
TOP_K = 20
TOP_P = 0.3
ALPHA_F = 0.6
ALPHA_P = 0.6

last_seen_toks = []
def prefill(model, toks, device, start_pos=0):
   global last_seen_toks

   # we can skip part of the prompt if it is the same as last and start_pos=0
   if start_pos == 0:
      for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
         if a != b: break
      else: i = min(len(toks), len(last_seen_toks))
      start_pos += i
      last_seen_toks = toks
      toks = toks[i:]

   # prefill the model
   for tok in tqdm(toks):
      GlobalCounters.reset()
      model(Tensor([[tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
      start_pos += 1
   return start_pos


if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("Performs a test load and generation of the specified model")
   parser.add_argument('model', choices=list(MODELS.keys()), help="Which model to load and test")
   parser.add_argument('--gpus', type=int, default=4, help="Number of GPUs allowed to be used")
   parser.add_argument('--dev-offset', type=int, default=1, help="Skips the first N devices to load the weights")
   parser.add_argument('--vram-limit', type=int, default=16, help="Amount of VRAM in GB per GPU to load weights")

   parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
   parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
   parser.add_argument("--prompt", type=str, default="Q: What is the distance from the earth to the moon?\nA: ", help="Phrase to start with")
   parser.add_argument("--timing", action="store_true", help="Print timing per token")
   parser.add_argument("--skip-load", action="store_true")
   
   args = parser.parse_args()

   # Load model
   with Context(BEAM=0):
      inst = MODELS[args.model]
      device_mem: Dict[str,int] = {}
      for i in range(args.dev_offset, args.dev_offset + args.gpus):
         device_mem[f"{Device.DEFAULT}:{i}"] = args.vram_limit*(1024**3)
      devices = tuple(device_mem.keys())
      model, tokenizer = load_model(inst, device_mem, args.skip_load)

      param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(model))

   def encode(content:str):
      return tokenizer.encode(content.strip(), add_special_tokens=False)

   MAX_TOKENS = 256

   user_start_token = encode(user_start_text := "[INST]")
   user_end_token   = encode(user_end_text   := "[/INST]")
   bos_token = encode(bos_text := tokenizer.special_tokens_map["bos_token"])
   eos_token = encode(eos_text := tokenizer.special_tokens_map["eos_token"])

   print(f"{user_start_text=}, {user_start_token=}")
   print(f"{user_end_text=}, {user_end_token=}")
   print(f"{bos_text=}, {bos_token=}")
   print(f"{eos_text=}, {eos_token=}")

   # for text in [user_role, assistant_role, thinking_tag, "Test", "Hello, world!"]:
   #    print(f"{text}: {encode(text)}")

   prompt = encode(f"{bos_text}You are an helpful assistant.")

   assert not args.skip_load
   start_pos = prefill(model, prompt, devices)
   while True:
      prompt = "Q: "
      # user_input = input(prompt).strip()
      # print()
      user_input = "What is the distance between the earth and the moon?"
      print(user_input)
      toks = encode(user_start_text + prompt + user_input + user_end_text)

      start_pos = prefill(model, toks[:-1], devices, start_pos=start_pos)
      last_tok = toks[-1]
      count = 0
      is_thinking = True
      while True:
         GlobalCounters.reset()
         if args.timing: print("")
         st = GlobalCounters.time_sum_s
         with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                     f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                     (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):

               tok = model(Tensor([[last_tok]], device=devices), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
         tok = tok.item()
         start_pos += 1
         count += 1
         last_tok = tok
         if tok in eos_token or count >= MAX_TOKENS:
            break
         print(tokenizer.decode([tok]), end="", flush=True)
      print(flush=True)
      break
