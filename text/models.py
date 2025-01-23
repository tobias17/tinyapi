from tinygrad import Device, nn
from tinygrad.helpers import fetch

from extra.models.llama import Transformer

from dataclasses import dataclass
from typing import Dict, Tuple

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

def load_model(arch:ModelArch, inst:ModelInst, device_mem:Dict[str,int]):
   index_filepath = fetch(f"{inst.weights_url}/{inst.index_filename}?download=true", "model.safetensors.index.json", subdir=inst.weights_subdir)
   for i in range(1, inst.num_weights+1):
      filename = inst.chunk_filename.format(i=i, num_weights=inst.num_weights)
      fetch(f"{inst.weights_url}/{filename}?download=true", filename, subdir=inst.weights_subdir)
   
   model = Transformer(**inst.params, linear=nn.Linear)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("Performs a test load and generation of the specified model")
   model_options: Dict[str,Tuple[ModelArch,ModelInst]] = {}
   for arch_name, arch_val in MODELS.items():
      for inst_name, inst_val in arch_val.instances.items():
         model_options[f"{arch_name}-{inst_name}"] = (arch_val,inst_val)
   parser.add_argument('model', choices=list(model_options.keys()), help="Which model to load and test")
   parser.add_argument('--gpus', type=int, default=6, help="Number of GPUs allowed to be used")
   parser.add_argument('--vram-limit', type=int, default=18, help="Amount of VRAM in GB per GPU to load weights")
   args = parser.parse_args()

   arch, inst = model_options[args.model]
   device_mem: Dict[str,int] = {}
   for i in range(args.gpus):
      device_mem[f"{Device.DEFAULT}:{i}"] = args.vram_limit*1024*1024*1024
   load_model(arch, inst, device_mem)
