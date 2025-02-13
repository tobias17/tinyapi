from tinygrad import Tensor, Device, Context, GlobalCounters
from transformers import PreTrainedTokenizer
from bottle import Bottle, request, response, HTTPResponse, abort, static_file
from typing import Union, Tuple, List, Dict
from pathlib import Path
import json, random, time, traceback

from models import MODELS, TokenSampler, build_transformer

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser("ModelTest", description="Performs a test load and generation of the specified model")
   parser.add_argument('model', choices=list(MODELS.keys()), help="Which model to load and test")
   parser.add_argument('--host', type=str, default="0.0.0.0", help="Web server bind address")
   parser.add_argument('--port', type=int, default=7776, help="Web server port")
   parser.add_argument("--debug", action="store_true", help="Enable debug mode")
   parser.add_argument('--gpus', type=int, default=4, help="Number of GPUs allowed to be used")
   parser.add_argument('--gpu-offset', type=int, default=1, help="Skips the first N devices to load the weights")
   parser.add_argument("--skip-load", action="store_true")
   args = parser.parse_args()

   Tensor.manual_seed(42)
   Tensor.no_grad = True

   # Load model
   with Context(BEAM=0):
      arch = MODELS[args.model]
      device: Union[str,Tuple[str,...]] = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.gpu_offset, args.gpu_offset+args.gpus))
      if len(device) == 1:
         device = device[0]

      model, tokenizer = build_transformer(arch, device) if not args.skip_load else ((lambda a,b,c: 1), PreTrainedTokenizer())

   # Prepare promtps and encoding
   p = arch.prompt.sub_vars(tokenizer)
   assert len(p.eos_tokens) > 0
   def encode_messages(messages:List[Dict]) -> List[int]:
      text = ""
      for msg in messages:
         if msg["role"] == "system":
            text += p.system_message.format(msg["content"])
         elif msg["role"] == "user":
            text += p.user_message.format(msg["content"])
         elif msg["role"] == "assistant":
            text += p.assistant_prefix + msg["content"] + p.assistant_suffix
         else:
            raise KeyError(f"Got unknown role key '{msg['role']}'")
      print(f"\nEncoding: |{text}|\n")
      return tokenizer.encode(text, add_special_tokens=False)

   SAMPLER = TokenSampler(
      temperature=0.95,
      top_k=0,
      top_p=0.0,
      alpha_f=0.0,
      alpha_p=0.0,
   )
   tokens = tokenizer.encode(p.system_message.format(arch.default_system_prompt), add_special_tokens=False)
   print("Preloading model and JIT")
   model(tokens, device, SAMPLER)

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
   def handle_500_error(error):
      response.content_type = 'text/plain'
      print(error)
      return 'Error 500: Internal Server Error\n\n'

   root = Path(__file__).parent/"../tinygrad/examples/tinychat"
   @app.route("/<filename>")
   def server_static(filename): return static_file(filename, root=root.as_posix())
   @app.route("/assets/<filename:path>")
   def server_assets(filename): return static_file(filename, root=(root/"assets").as_posix())
   @app.route("/")
   def index():
      return static_file("index.html", root=root.as_posix())

   @app.get("/v1/models")
   def models() -> str:
      return json.dumps([str(args.model)])

   @app.post("/v1/internal/token-count")
   def token_count() -> str:
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))
   @app.post("/v1/token/encode")
   def token_encode() -> str:
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", "")))

   # @app.post("/v1/completions")
   # def completions():
   #    rjson = json.loads(request.body.read())

   #    # check if we are streaming
   #    if rjson.get("stream", False):
   #       response.content_type = "text/event-stream"
   #       response.set_header("Cache-Control", "no-cache")
   #    else: abort(400, "streaming required")

   #    toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)

   #    start_pos = prefill(model, toks[:-1])
   #    last_tok = toks[-1]
   #    while True:
   #       GlobalCounters.reset()
   #       tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
   #       start_pos += 1
   #       last_tok = tok
   #       if tok in p.eos_tokens: break

   #       res = {
   #          "choices": [{
   #             "text": tokenizer.decode([tok]),
   #          }]
   #       }
   #       yield f"data: {json.dumps(res)}\n\n"

   @app.post("/v1/chat/token/encode")
   def chat_token_encode() -> str:
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")
      tokens = encode_messages(rjson["messages"])
      return json.dumps(tokens)

   @app.post("/v1/chat/completions")
   def chat_completions():
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")

      # check if we are streaming
      if rjson.get("stream", False):
         response.content_type = "text/event-stream"
         response.set_header("Cache-Control", "no-cache")
      else: abort(400, "streaming required")
      random_id = random.randbytes(16).hex()

      messages = [m for m in rjson["messages"]]
      if messages[0]["role"] != "system":
         messages.insert(0, {"role":"system", "content":arch.default_system_prompt})
      tokens = encode_messages(messages)

      while True:
         GlobalCounters.reset()
         tok = model(tokens, device, SAMPLER)
         tokens.append(tok)
         if tok in p.eos_tokens: break

         res = {
            "id": random_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(args.model),
            "choices": [{
               "index": 0,
               "delta": {
                  "role": "assistant",
                  "content": tokenizer.decode([tok]),
               },
               "finish_reason": None,
            }]
         }
         yield f"data: {json.dumps(res)}\n\n"

      res = {
         "id": random_id,
         "object": "chat.completion.chunk",
         "created": int(time.time()),
         "model": str(args.model),
         "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
         }]
      }
      yield f"data: {json.dumps(res)}\n\n"

   app.run(host=args.host, port=args.port, debug=True)
