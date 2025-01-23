from extra.models.llama import Transformer
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ModelInst:
   params: Dict
   weights_url: str
   num_weights: int

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
            num_weights=17,
         ),
      }
   )
}

def load_model(arch:ModelArch, inst:ModelInst):
   pass

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("Performs a test load and generation of the specified model")
   model_options: Dict[str,Tuple[ModelArch,ModelInst]] = {}
   for arch_name, arch_val in MODELS.items():
      for inst_name, inst_val in arch_val.instances.items():
         model_options[f"{arch_name}-{inst_name}"] = (arch_val,inst_val)
   parser.add_argument('model', choices=list(model_options.keys()))
   args = parser.parse_args()

   arch, inst = model_options[args.model]
   load_model(arch, inst)
