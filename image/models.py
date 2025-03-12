from tinygrad import Tensor, Device, dtypes, Context, TinyJit
from tinygrad.helpers import fetch, tqdm
from tinygrad.nn.state import load_state_dict, safe_load, get_parameters, get_state_dict
from examples.sdxl import SDXL, DPMPP2MSampler, configs
from typing import Tuple, Union
from PIL import Image
import re

class Defaults:
   WIDTH    = 768
   HEIGHT   = 1024
   GUIDANCE = 7.0
   STEPS    = 20

MODEL_NAME, MODEL_URL = 'juggernaut_xl.safetensors', 'https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'

def remap_lora_weight_name(name:str) -> str:
   name = re.sub(r'_', '.', name)
   name = re.sub(r'lora\.unet\.', 'model.diffusion_model.', name)
   name = re.sub(r'put\.blocks', 'put_blocks', name)
   name = re.sub(r'\.middle\.block\.', '.middle_block.', name)
   name = re.sub(r'\.to\.', '.to_', name)
   name = re.sub(r'\.transformer\.blocks\.', '.transformer_blocks.', name)
   name = re.sub(r'\.proj\.in\.', '.proj_in.', name)
   name = re.sub(r'\.proj\.out\.', '.proj_out.', name)
   return name

def load_lora_onto_(model:SDXL) -> None:
   model_state_dict = get_state_dict(model)
   lora_state_dict  = safe_load(fetch("https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors", "pixel-art-xl.safetensors"))

   remapped_state_dict = {}
   for k, w in lora_state_dict.items():
      remapped_state_dict[remap_lora_weight_name(k)] = w

   seen_names = set()
   for mapped_name, w in remapped_state_dict.items():
      if mapped_name.endswith(".alpha"):
         continue
      base_name = re.sub(r'\.lora\.(down|up)', '', mapped_name)
      if base_name in seen_names:
         continue
      seen_names.add(base_name)

      model_weight = model_state_dict.get(base_name, None)
      if model_weight is None:
         print(f"PANIC: missing model weight for lora application: {base_name}")
         continue

      try:
         def move(x:Tensor) -> Tensor: return x.to(model_weight.device).cast(model_weight.dtype)
         up_weight   = move(remapped_state_dict[re.sub(r'\.lora\.(down|up)',   '.lora.up',   mapped_name)])
         down_weight = move(remapped_state_dict[re.sub(r'\.lora\.(down|up)',   '.lora.down', mapped_name)])
         alpha       = move(remapped_state_dict[re.sub(r'\.lora\.(down|up).+', '.alpha',     mapped_name)])

         lora_contribution = (up_weight @ down_weight) * (alpha / down_weight.shape[0])
         model_weight.replace((model_weight + lora_contribution).realize())
      except KeyError as ex:
         print(f"Failed when running mapped_name: {mapped_name}")
         print(f"Failed when running base_name:   {base_name}")
         raise ex from ex

def load_sdxl(device:Union[str,Tuple[str,...]], guidance_scale:float=Defaults.GUIDANCE) -> Tuple[SDXL,DPMPP2MSampler]:
   model = SDXL(configs["SDXL_Base"])
   weights = fetch(MODEL_URL, MODEL_NAME)

   with Context(BEAM=0):
      assert isinstance(device, str), f"Multi device image generation not yet supported"
      for w in get_parameters(model):
         w.to_(device)

      load_state_dict(model, safe_load(weights), strict=False)
      load_lora_onto_(model)

   return model, DPMPP2MSampler(guidance_scale)

@TinyJit
def make_noise_image(shape:Tuple[int,...]) -> Tensor:
   return Tensor.randn(shape).realize()

@TinyJit
def decode_step(model:SDXL, z:Tensor) -> Tensor:
   return model.decode(z).realize()

def generate_image(model:SDXL, sampler:DPMPP2MSampler, prompt:str, img_width:int=Defaults.WIDTH, img_height:int=Defaults.HEIGHT, num_steps:int=Defaults.STEPS, warmup_decoder:bool=False) -> Image.Image:
   N = 1
   C = 4
   F = 8

   assert img_width  % F == 0, f"img_width must be multiple of {F}, got {img_width}"
   assert img_height % F == 0, f"img_height must be multiple of {F}, got {img_height}"

   c, uc = model.create_conditioning([prompt], img_width, img_height)
   for v in c .values(): v.realize()
   for v in uc.values(): v.realize()

   shape = (N, C, img_height // F, img_width // F)
   if warmup_decoder:
      for _ in range(2):
         make_noise_image(shape)
   randn = make_noise_image(shape)

   z = sampler(model.denoise, randn, c, uc, num_steps)
   if warmup_decoder:
      print("Warming up decoder")
      for i in tqdm(range(3)):
         decode_step(model, (z * (i+2)).realize())
   x = decode_step(model, z.realize())
   x = (x + 1.0) / 2.0
   x = x.reshape(3,img_height,img_width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)

   return Image.fromarray(x.numpy())

if __name__ == "__main__":
   model, sampler = load_sdxl(Device.DEFAULT)
   im = generate_image(model, sampler, "a horse sized cat eating a bagel, pixel art", warmup_decoder=False)
   im.save("/tmp/gen.png")
