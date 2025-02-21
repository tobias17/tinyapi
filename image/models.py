from tinygrad import Tensor, Device, dtypes, Context, TinyJit
from tinygrad.helpers import fetch, tqdm
from tinygrad.nn.state import load_state_dict, safe_load, get_parameters
from typing import Tuple, Union
from PIL import Image

from examples.sdxl import SDXL, DPMPP2MSampler, configs

class Defaults:
   WIDTH    = 768
   HEIGHT   = 1024
   GUIDANCE = 7.0
   STEPS    = 20

def load_sdxl(device:Union[str,Tuple[str,...]], guidance_scale:float=Defaults.GUIDANCE) -> Tuple[SDXL,DPMPP2MSampler]:
   model = SDXL(configs["SDXL_Base"])
   default_weight_url = 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors'
   weights = fetch(default_weight_url, 'sd_xl_base_1.0.safetensors')

   with Context(BEAM=0):
      assert isinstance(device, str), f"Multi device image generation not yet supported"
      for w in get_parameters(model):
         w.to_(device)

      load_state_dict(model, safe_load(weights), strict=False)

   return model, DPMPP2MSampler(guidance_scale)

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
   randn = Tensor.randn(shape)

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
   im = generate_image(model, sampler, "a horse sized cat eating a bagel", warmup_decoder=True)
   im.save("/tmp/gen.png")
