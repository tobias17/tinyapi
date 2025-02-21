from tinygrad import Tensor, Context, GlobalCounters
from pathlib import Path
from typing import Tuple, List, Dict
import json, random, time

from bottle import Bottle, request, response, abort, static_file # type: ignore
from text.models import MODELS, TokenSampler, build_transformer


MAX_NEW_TOKENS = 512


def add_text_endpoints(app:Bottle, model_name:str, device:Tuple[str,...]) -> None:
   # Load model
   with Context(BEAM=0):
      arch = MODELS[model_name]
      model, tokenizer = build_transformer(arch, device)

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
      return tokenizer.encode(text, add_special_tokens=False)

   SAMPLER = TokenSampler(
      temperature=0.7,
      top_k=0,
      top_p=0.0,
      alpha_f=0.0,
      alpha_p=0.0,
   )
   tokens = tokenizer.encode(p.system_message.format(arch.default_system_prompt), add_special_tokens=False)
   print("Warming up text model")
   model(tokens, device, SAMPLER)


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
      return json.dumps([str(model_name)])

   @app.post("/v1/internal/token-count")
   def token_count() -> str:
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))
   @app.post("/v1/token/encode")
   def token_encode() -> str:
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", "")))

   @app.post("/v1/chat/token/encode")
   def chat_token_encode() -> str:
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")
      tokens = encode_messages(rjson["messages"])
      return json.dumps(tokens)

   @app.post("/v1/chat/completions")
   def chat_completions():
      Tensor.manual_seed(time.time())
      rjson = json.loads(request.body.read())
      if "messages" not in rjson: abort(400, "messages required")

      # check if we are streaming
      streaming = rjson.get("stream", False)
      if streaming:
         response.content_type = "text/event-stream"
         response.set_header("Cache-Control", "no-cache")
      random_id = random.randbytes(16).hex()

      messages = [m for m in rjson["messages"]]
      if messages[0]["role"] != "system":
         messages.insert(0, {"role":"system", "content":arch.default_system_prompt})
      tokens = encode_messages(messages)
      new_tokens = []

      count = 0
      while True:
         count += 1
         if count >= MAX_NEW_TOKENS:
            new_tokens.append(tokenizer.encode(p.assistant_suffix, add_special_tokens=False))
            break

         GlobalCounters.reset()
         tok = model(tokens, device, SAMPLER)
         tokens.append(tok)
         if tok in p.eos_tokens: break
         new_tokens.append(tok)

         if streaming:
            res = {
               "id": random_id,
               "object": "chat.completion.chunk",
               "created": int(time.time()),
               "model": str(model_name),
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
         "model": str(model_name),
         "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
         }]
      } if streaming else {
         "id": random_id,
         "object": "chat.completion",
         "created": int(time.time()),
         "choices": [{
            "index": 0,
            "message": {
               "role": "assistant",
               "content": tokenizer.decode(new_tokens),
            },
         }]
      }
      yield f"data: {json.dumps(res)}\n\n"
