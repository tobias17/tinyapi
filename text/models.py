from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.helpers import fetch, tqdm
from tinygrad.nn.state import get_state_dict, load_state_dict
from examples.llama3 import load

from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16

from typing import Dict, Tuple, Set, List
from dataclasses import dataclass

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

def split_into_blocks(state_dict:Dict[str,Tensor], n_layers:int) -> List[List[Tensor]]:
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

   return blocks

def load_model(inst:ModelInst, device_mem:Dict[str,int]) -> Transformer:
   TARGET_DTYPE = dtypes.float16

   index_filename = "model.safetensors.index.json"
   model_path = fetch(f"{inst.weights_url}/{inst.index_filename}?download=true", index_filename, subdir=inst.weights_subdir)
   for i in range(1, inst.num_weights+1):
      filename = inst.chunk_filename.format(i=i, num_weights=inst.num_weights)
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)
   
   model = Transformer(**inst.params, linear=nn.Linear)
   weights = fix_bf16(convert_from_huggingface(load(str(model_path)), model, inst.params["n_heads"], inst.params["n_kv_heads"], permute_layers=False))

   mem_rem = {k:v for k,v in device_mem.items()}
   dev_key = list(mem_rem.keys())[0]
   blocks = split_into_blocks(get_state_dict(model), inst.params["n_layers"])
   for b in tqdm(blocks):
      sz = sum(w.lazydata.size * TARGET_DTYPE.itemsize for w in b)
      while True:
         if sz <= mem_rem[dev_key]:
            for w in b:
               w.replace(w.to(dev_key).cast(TARGET_DTYPE).realize())
            mem_rem[dev_key] -= sz
            break

         mem_rem.pop(dev_key)
         assert len(mem_rem) > 0, f"Ran out of memory loading model"
         dev_key = list(mem_rem.keys())[0]

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
   parser.add_argument('--gpus', type=int, default=5, help="Number of GPUs allowed to be used")
   parser.add_argument('--dev-offset', type=int, default=1, help="Skips the first N devices to load the weights")
   parser.add_argument('--vram-limit', type=int, default=16, help="Amount of VRAM in GB per GPU to load weights")
   args = parser.parse_args()

   arch, inst = model_options[args.model]
   device_mem: Dict[str,int] = {}
   for i in range(args.dev_offset, args.dev_offset + args.gpus):
      device_mem[f"{Device.DEFAULT}:{i}"] = args.vram_limit*(1024**3)
   load_model(inst, device_mem)
