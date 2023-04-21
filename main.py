import re
import sys
import asyncio
import uvicorn
from typing import Union
from pathlib import Path
from modules import shared
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# Ooba imports:
from modules.models import load_model
from modules.LoRA import add_lora_to_model
from modules.models import clear_torch_cache
from modules.text_generation import encode, generate_reply, stop_everything_event


def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)


# Setup FastAPI:
app = FastAPI()
semaphore = asyncio.Semaphore(1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    skip_special_tokens: Optional[bool] =False
    streaming: Optional[bool] =True
    # task_id?


@app.get("/")
def hellow_world(q: Union[str, None] = None):
    return {"wintermute": "ai", "q": q}


# Webserver uses this to check if LLM is running:
@app.get("/check")
def check():
    return shared.model_name


# in generate strip to the last . rather than ending in the middle of a sentence. (?)
@app.post("/generate")
async def stream_data(req: GenerateRequest):
    while True:
        try:
            # Attempt to acquire the semaphore without waiting
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            print("Server is busy")
            await asyncio.sleep(1)
            #raise HTTPException(status_code=503, detail="Server is busy, please try again later")

    try:
        print(req.prompt)
  
        prompt_lines = [k.strip() for k in req.prompt.split('\n')]
        req.prompt = '\n'.join(prompt_lines)

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
            'skip_special_tokens': req.skip_special_tokens,
            'streaming': req.streaming
        }
        #print(generate_params)

        # Hooking no_stream arg so that we can set streaming value from here:
        if req.streaming:
            shared.args.no_stream = False
        else:
            shared.args.no_stream = True

        # Set custom stopping strings from req.
        if req.custom_stopping_strings!="":
            stop = [req.custom_stopping_strings]
        else:
            stop = []
    
        # start generating response:
        generator = generate_reply(
            req.prompt, #question
            generate_params, #state
            eos_token=None,
            stopping_strings= stop,
        )

        async def gen():
            _len = 0
            answer = ""
            answer_str = ""
            last_answer = ""
            for a in generator:
                if isinstance(a, str):
                    answer = a
                else:
                    answer = a[0]

                # remove prompt from response:
                answer = answer.replace(req.prompt,"")

                # remove last part of the stream from response:
                _answ = answer[_len:]
                #print("a: {0}".format(_answ), flush=True)

                # set next last_answer:
                last_answer = answer
                _len = len(last_answer)

                yield _answ.encode("utf-8")

        return StreamingResponse(gen())
    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()

if __name__ == "__main__":
    # get available models:
    available_models = get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            print('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')
            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()
        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        shared.model, shared.tokenizer = load_model(shared.model_name)
        #add lora
        if shared.args.lora:
            add_lora_to_model([shared.args.lora])

    #python main.py --model-menu --model_type llama --wbits 4 --groupsize 128 --xformers
    #python main.py --model-menu --model_type llama --load-in-8bit --lora baize-lora-13B --xformers
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7861
    )