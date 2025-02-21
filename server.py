from tinygrad import Tensor, Device
from bottle import Bottle, request, response, HTTPResponse, HTTPError
from typing import Optional

from text.models import MODELS as TEXT_MODELS
from text.endpoint import add_text_endpoints

def run_server(host:str, port:int, text_model:Optional[str], image_model:Optional[str]) -> None:
   assert any(m is not None for m in (text_model, image_model)), f"Got no models to load"
   Tensor.no_grad = True

   app = Bottle()
   cors_headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
      "Access-Control-Allow-Credentials": "true",
   }
   @app.hook("before_request")
   def handle_options():
      if request.method == "OPTIONS": raise HTTPResponse(headers=cors_headers)
   @app.hook("after_request")
   def enable_cors():
      for key, value in cors_headers.items(): response.set_header(key, value)
   @app.error(500)
   def handle_500_error(error:HTTPError):
      response.content_type = 'text/plain'
      print(error.traceback)
      return f'Error 500: Internal Server Error\n\n{error.traceback}'

   if text_model is not None:
      # TODO: make device assignment better
      add_text_endpoints(app, text_model, tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5)))

   if image_model is not None:
      # TODO: make device assignment better
      pass

   app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("ModelTest", description="Performs a test load and generation of the specified model")
   parser.add_argument('--text-model',  choices=list(TEXT_MODELS.keys()), help="Which text model to load and serve")
   parser.add_argument('--image-model')
   parser.add_argument('--host', type=str, default="0.0.0.0", help="Web server bind address")
   parser.add_argument('--port', type=int, default=7776, help="Web server port")
   args = parser.parse_args()

   run_server(args.host, args.port, args.text_model, args.image_model)
