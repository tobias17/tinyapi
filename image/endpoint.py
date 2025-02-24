from tinygrad import Context
from typing import Tuple, Union
from io import BytesIO
from PIL import Image
import json, base64, time

from bottle import Bottle, request, response, abort, static_file # type: ignore
from image.models import load_sdxl, generate_image

BEAM_VALUE = 1

DEFAULT_IMAGE_FILETYPE = "JPEG"

def base64_encode(im:Image.Image) -> str:
   buffered = BytesIO()
   im.save(buffered, format=DEFAULT_IMAGE_FILETYPE)
   return base64.b64encode(buffered.getvalue()).decode()

def add_image_endpoints(app:Bottle, model_name:str, device:Union[str,Tuple[str,...]]) -> None:
   # Load model
   with Context(BEAM=0):
      model, sampler = load_sdxl(device)

   # Warmup
   print("Warming up sampler")
   with Context(BEAM=BEAM_VALUE):
      generate_image(model, sampler, "a horse sized cat eating a bagel", warmup_decoder=True)

   @app.post("/v1/txt2img")
   def models() -> str:
      s_time = time.time()

      rjson = json.loads(request.body.read())
      prompt = rjson.get("prompt", None)
      if prompt is None:
         abort(400, "prompt required")

      with Context(BEAM=BEAM_VALUE):
         im = generate_image(model, sampler, prompt)
      res = {
         "image": base64_encode(im),
         "filetype": DEFAULT_IMAGE_FILETYPE,
      }

      print(f"Generated image in {time.time() - s_time:.1f} seconds")
      return f"data: {json.dumps(res)}\n\n"
