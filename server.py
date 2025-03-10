from tinygrad import Tensor
from bottle import Bottle, request, response, HTTPResponse, HTTPError # type: ignore
from typing import Dict
import json, os

from text.models import MODELS as TEXT_MODELS
from text.endpoint import add_text_endpoints
from image.endpoint import add_image_endpoints

def run_server(host:str, port:int, text_cfg:Dict, image_cfg:Dict) -> None:
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

   if text_cfg.get("enabled", True):
      add_text_endpoints(app, text_cfg)

   if image_cfg.get("enabled", True):
      add_image_endpoints(app, image_cfg)

   app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("ModelTest", description="Performs a test load and generation of the specified model")
   parser.add_argument('config_file', type=str, help="Path to json file containing the server config")
   args = parser.parse_args()

   assert os.path.exists(args.config_file), f"Could not find config file, searched for {os.path.abspath(args.config_file)}"
   with open(args.config_file) as f:
      cfg = json.load(f)
   
   host  = cfg.get("host", "0.0.0.0")
   port  = cfg.get("port", 7776)
   text  = cfg.get("text",  {"enabled":False})
   image = cfg.get("image", {"enabled":False})

   run_server(host, port, text, image)
