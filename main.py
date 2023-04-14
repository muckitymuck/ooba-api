import re
import sys
import json
import time
import torch
import transformers
from pathlib import Path
from modules import shared
from modules.models import load_model
from modules.LoRA import add_lora_to_model
from modules.models import clear_torch_cache
from modules.text_generation import encode, generate_reply, stop_everything_event

import uvicorn
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, Request
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
#from sse_starlette.sse import EventSourceResponse


# can use this in a fastAPI call to change lora?
def load_lora_wrapper(selected_lora):
    add_lora_to_model(selected_lora)
    return selected_lora

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

# Llama-Precise:
    #do_sample=True
    #top_p=0.1
    #top_k=40
    #temperature=0.7
    #repetition_penalty=1.18
    #typical_p=1.0

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 200
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    typical_p: Optional[float] = 1
    repetition_penalty: Optional[float] = 1.18
    encoder_repetition_penalty: Optional[float] = 1
    top_k: Optional[float] = 40
    min_length: Optional[int] = 0
    no_repeat_ngram_size: Optional[float] = 0 #int?
    num_beams: Optional[int] = 1
    penalty_alpha: Optional[float] = 0 #int
    length_penalty: Optional[float] = 1 #int
    early_stopping: Optional[bool] = False
    seed: Optional[int] = -1
    #n: Optional[int] = None
    stream: Optional[bool] = False
    return_prompt: Optional[bool] = False
    #stop: Optional[str] = None
    add_bos_token: Optional[bool] = True
    truncation_length: Optional[int] = 2048
    custom_stopping_strings: Optional[str] = ''
    ban_eos_token: Optional[bool] =False

@app.get("/")
def hellow_world(q: Union[str, None] = None):
    return {"wintermute": "ai", "q": q}


# in generate strip to the last . rather than ending in the middle of a sentence. (?)
@app.post("/generate")
async def generate(req: GenerateRequest):
    print(req.prompt)
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{0}
### Response:
""".format(req.prompt)
    #print(prompt)

    prompt_lines = [k.strip() for k in prompt.split('\n')]

    #max_context = 2048
    # enforce context length...
    #while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
    #    prompt_lines.pop(0)

    prompt = '\n'.join(prompt_lines)

    # need to allow over-ride from params passed in from user, lol
    generate_params = {
        'max_new_tokens': req.max_new_tokens,
        'do_sample': req.do_sample,
        'temperature': req.temperature,
        'top_p': req.top_p,
        'typical_p': req.typical_p,
        'repetition_penalty': req.repetition_penalty,
        'encoder_repetition_penalty': req.encoder_repetition_penalty,
        'top_k': req.top_k,
        'min_length': req.min_length,
        'no_repeat_ngram_size': req.no_repeat_ngram_size,
        'num_beams': req.num_beams,
        'penalty_alpha': req.penalty_alpha,
        'length_penalty': req.length_penalty,
        'early_stopping': req.early_stopping,
        'seed': req.seed,
        'add_bos_token': req.add_bos_token,
        'truncation_length': req.truncation_length,
        'custom_stopping_strings': req.custom_stopping_strings,
        'ban_eos_token': req.ban_eos_token,
    }
    #print(generate_params)

    #def generate_reply(question, state, eos_token=None, stopping_strings=[]):
    generator = generate_reply(
        prompt,
        generate_params,
        eos_token=None,
        stopping_strings=[],
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


    #self.wfile.write(response.encode('utf-8'))
    #print(response.encode('utf-8'))
    print(answer[len(req.prompt):])
    print(answer[:len(req.prompt)])
    print(answer.replace(req.prompt,""))

    return {"wintermute": "ai", "response": answer[len(req.prompt):]}


@app.get("/check")
def check():
    return "Ok"

if __name__ == "__main__":
    # model check:
    model_check()

    # load model
    shared.model, shared.tokenizer = load_model(shared.model_name)

    if shared.args.lora:
        add_lora_to_model(shared.args.lora)

    #python main.py --model_type llama --wbits 4 --groupsize 128 --xformers
    #python main.py --model_type llama --load-in-8bit --lora baize-lora-13B --xformers
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7861
    )