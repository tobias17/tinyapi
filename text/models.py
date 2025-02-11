from tinygrad import Tensor, Device, dtypes, nn, Context
from tinygrad.helpers import fetch, colored, GlobalCounters, Timing, DEBUG, tqdm
from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from examples.llama3 import load

from extra.models.llama import Transformer, ModelConfig, convert_from_huggingface, fix_bf16, TokenSampler

from typing import Dict, Tuple, Set, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer
from pathlib import Path
import re, json, os

variable_pattern = re.compile(r"%%([^%]+)%%")
@dataclass
class Prompt:
   system_message: str
   user_message: str
   assistant_prefix: str
   assistant_suffix: str

   def sub_vars(self, tokenizer:AutoTokenizer) -> None:
      for name, template in asdict(self).items():
         while True:
            m = variable_pattern.search(template)
            if not m: break
            value = tokenizer.special_tokens_map.get(key := m.group(1), None)
            assert value is not None, f"Failed to find key '{key}' in special_tokens_map, options were {list(tokenizer.special_tokens_map.keys())}"
            template = template.replace(f"%%{key}%%", value)
         setattr(self, name, template)

@dataclass
class ModelInst:
   config: ModelConfig
   weights_url: str
   weights_subdir: str
   num_weights: int
   prompt: Prompt
   index_filename: str = "model.safetensors.index.json"
   chunk_filename: str = "model-{i:05d}-of-{num_weights:05d}.safetensors"
   extra_filenames: List[str] = field(default_factory=lambda: ["tokenizer.json", "tokenizer_config.json"])
   fix_weights: Callable[[Transformer,ModelConfig],Transformer] = (lambda m,c: m)


def fix_qwen_weights(model:Transformer, cfg:ModelConfig) -> Transformer:
   updated_layers = []
   for layer in model.layers:
      head_dim = cfg.get_head_dim()
      layer.attention.wq = nn.Linear(cfg.dim, cfg.n_heads    * head_dim, bias=True)
      layer.attention.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * head_dim, bias=True)
      layer.attention.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * head_dim, bias=True)
      updated_layers.append(layer)
   model.layers = updated_layers
   return model


TARGET_DTYPE = dtypes.float16
MODELS: Dict[str,ModelInst] = {
   # "Qwen-32B": ModelInst(
   #    params={"dim":5120, "n_heads":40, "n_kv_heads":8, "n_layers":64, "norm_eps":1e-5, "rope_theta":1000000, "vocab_size":152064, "hidden_dim":27648},
   #    weights_url="https://huggingface.co/Qwen/QwQ-32B-Preview/resolve/main",
   #    weights_subdir="qwq_32b_preview",
   #    num_weights=17,
   # ),
   "Qwen-R1-32B": ModelInst(
      config=ModelConfig(dim=5120, hidden_dim=27648, n_layers=64, n_heads=40, n_kv_heads=8, norm_eps=1e-5, vocab_size=152064, rope_theta=1000000.0, max_context=4096),
      weights_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/resolve/main",
      weights_subdir="deepseek_r1_qwen_32b",
      num_weights=8,
      chunk_filename="model-{i:05d}-of-{num_weights:06d}.safetensors", # WHY????
      # Prompt template: https://unsloth.ai/blog/deepseekr1-dynamic
      fix_weights=fix_qwen_weights,
      prompt=Prompt("%%bos_token%%{0}", "<｜User｜>{0}\n", "<｜Assistant｜><think></think>", "\n"),
   ),
   "Mistral-24B": ModelInst(
      config=ModelConfig(dim=5120, hidden_dim=32768, n_layers=40, n_heads=32, head_dim=128, n_kv_heads=8, norm_eps=1e-5, vocab_size=131072, rope_theta=100000000.0, max_context=32768),
      weights_url="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/resolve/main",
      weights_subdir="mistral_small_24b_instruct",
      num_weights=10,
      prompt=Prompt("<s>{0}\n\n", "[INST] {0} [/INST]", "", ""),
   ),
   "Llama-8B": ModelInst(
      config=ModelConfig(dim=4096, hidden_dim=14336, n_layers=32, n_heads=32, n_kv_heads=8, norm_eps=1e-5, vocab_size=128256, rope_theta=500000.0, max_context=4096),
      weights_url="https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main",
      weights_subdir="llama3-8b-sfr",
      num_weights=4,
      prompt=Prompt(
         "<|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>",
         "<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|>",
         "<|start_header_id|>assistant<|end_header_id|>\n\n",
         "<|eot_id|>",
      ),
   )
}

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

def diff_state_dict_keys(model_set:Set[str], disk_set:Set[str]) -> None:
   only_model = model_set.difference(disk_set)
   only_disk  = disk_set .difference(model_set)

   if len(only_model) > 0:
      print("\nONLY MODEL:\n" + "\n".join(only_model))
   if len(only_disk) > 0:
      print("\nONLY DISK:\n" + "\n".join(only_disk))
   if len(only_model) > 0 or len(only_disk) > 0:
      print()

def load_model_old(inst:ModelInst, device_mem:Dict[str,int], skip_load:bool=False) -> Tuple[Transformer,AutoTokenizer]:
   index_filename = "model.safetensors.index.json"
   model_path = fetch(f"{inst.weights_url}/{inst.index_filename}?download=true", index_filename, subdir=inst.weights_subdir)
   for i in range(1, inst.num_weights+1):
      filename = inst.chunk_filename.format(i=i, num_weights=inst.num_weights)
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)

   for filename in inst.extra_filenames:
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)
   tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent))

   model = Transformer(inst.config)
   model = inst.fix_weights(model, inst.config)

   shard_across(get_state_dict(model), tuple(device_mem.keys()))

   weights = fix_bf16(convert_from_huggingface(load(str(model_path)), model, inst.config.n_heads, inst.config.n_kv_heads, permute_layers=False))
   diff_state_dict_keys(set(get_state_dict(weights).keys()), set(weights.keys()))
   if not skip_load:
      load_state_dict(model, weights, strict=False, consume=True)

   return model, tokenizer




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
   import argparse
   parser = argparse.ArgumentParser("ModelTest", description="Performs a test load and generation of the specified model")
   parser.add_argument('model', choices=list(MODELS.keys()), help="Which model to load and test")
   parser.add_argument('--gpus', type=int, default=4, help="Number of GPUs allowed to be used")
   parser.add_argument('--dev-offset', type=int, default=1, help="Skips the first N devices to load the weights")
   parser.add_argument('--vram-limit', type=int, default=16, help="Amount of VRAM in GB per GPU to load weights")

   parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
   parser.add_argument("--prompt", type=str, default="Q: What is the distance from the earth to the moon?\nA: ", help="Phrase to start with")
   parser.add_argument("--timing", action="store_true", help="Print timing per token")
   parser.add_argument("--skip-load", action="store_true")

   args = parser.parse_args()

   Tensor.manual_seed(42)

   # Load model
   with Context(BEAM=0):
      inst = MODELS[args.model]
      # device_mem: Dict[str,int] = {}
      # for i in range(args.dev_offset, args.dev_offset + args.gpus):
      #    device_mem[f"{Device.DEFAULT}:{i}"] = args.vram_limit*(1024**3)
      # device = tuple(device_mem.keys())
      # if len(device) == 1:
      #    device = device[0]
      device = Device.DEFAULT

      # model, tokenizer = load_model(inst, device_mem, args.skip_load)
      model_path = fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json", "model.safetensors.index.json", subdir="llama3-8b-sfr")
      model = build_transformer(model_path, inst.config, 1, device=device)
      tokenizer = AutoTokenizer.from_pretrained(str(model_path.parent))

      param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(model))

   # Prepare promtps and encoding
   inst.prompt.sub_vars(tokenizer)
   def encode(content:str) -> List[int]:
      return tokenizer.encode(content, add_special_tokens=False)
   eos_token = encode(eos_text := tokenizer.special_tokens_map["eos_token"])

   MAX_TOKENS = 256
   SYSTEM_PROMPT = "You are an helpful assistant. Answer questions directly. Keep responses short and simple."
   LLAMA_TOKENS = [128000, 128006, 9125, 128007, 271, 2675, 527, 459, 11190, 18328, 13, 128009, 128006, 882, 128007, 271, 12840, 374, 279, 6138, 1990, 279, 9578, 323, 279, 18266, 30, 128009, 128006, 78191, 128007, 271]
   LLAMA_STOP   = {128001, 128009}

   sys_prompt = encode(full_system_prompt := inst.prompt.system_message.format(SYSTEM_PROMPT))
   print(full_system_prompt[:-1])
   assert not args.skip_load

   # prompt = "Q: "
   # user_input = inst.prompt.user_message.format("What is the distance between the earth and the moon?")
   # print(block := user_input + inst.prompt.assistant_prefix)
   # toks = sys_prompt + encode(block)
   toks = LLAMA_TOKENS

   count = 0
   is_thinking = True
   while True:
      tok = model(toks, device, SAMPLER)
      toks.append(tok)
      count += 1
      if tok in LLAMA_STOP or count >= MAX_TOKENS: break
      print(tokenizer.decode([tok]), end="", flush=True)
   print(flush=True)
