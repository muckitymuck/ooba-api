import re
import sys
import json
import time
import torch
import transformers
from pathlib import Path
from modules import shared
from modules.models import load_model
from modules.models import clear_torch_cache
from modules.text_generation import encode, generate_reply, stop_everything_event

import uvicorn
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, Request
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse


def model_check():
    # Get available models:
    available_models = sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

    # Default model
    if shared.args.model is not None:
        shared.model_name = shared.args.model
    else:
        if len(available_models) == 0:
            print('No models are available! Please download at least one.')
            sys.exit(0)
        elif len(available_models) == 1:
            i = 0
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')
            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()
        shared.model_name = available_models[i]

# Setup FastAPI:
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    #top_k: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[str] = None
    stream: Optional[bool] = False

@app.get("/")
def hellow_world(q: Union[str, None] = None):
    return {"wintermute": "ai", "q": q}


# ability to save in a local database for future training.
# in generate strip to the last . rather than ending in the middle of a sentence. (?)
@app.post("/generate")
async def generate(req: GenerateRequest):
    req.stream = False
    req.prompt = req.prompt
    req.temperature = req.temperature or 0.7
    req.stop = req.stop or "\n"

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{0}
### Response:
""".format(req.prompt)

    #print(req.prompt)
    #print("---")
    #print(prompt)
    #print("-------")

    prompt_lines = [k.strip() for k in prompt.split('\n')]
    max_context = 2048
    #max_context = body.get('max_context_length', 2048)

    # enforce context length...
    #while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
    #    prompt_lines.pop(0)

    prompt = '\n'.join(prompt_lines)
    #print(prompt)

    # Llama-Precise:
    #do_sample=True
    #top_p=0.1
    #top_k=40
    #temperature=0.7
    #repetition_penalty=1.18
    #typical_p=1.0

    # need to allow over-ride from params passed in from user, lol
    generate_params = {
        'max_new_tokens': 100,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'encoder_repetition_penalty': 1,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
    }

    generator = generate_reply(
        prompt,
        generate_params,
        stopping_strings=[],
        #stream=req.stream,
    )

    answer = ''
    last_answer = ''
    for a in generator:
        last_answer = answer

        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]

        #print(answer.replace(last_answer,""), end="", flush=True)

    response = json.dumps({
        'results': [{
            'output': answer
        }]
    })

    #self.wfile.write(response.encode('utf-8'))
    #print(response.encode('utf-8'))

    return {"wintermute": "ai", "response": answer}


@app.get("/check")
def check():
    return "Ok"

if __name__ == "__main__":
    # model check:
    model_check()

    # load model
    shared.model, shared.tokenizer = load_model(shared.model_name)

    #python server.py --auto-devices --wbits 4  --groupsize 128 --model_type llama
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7861
    )