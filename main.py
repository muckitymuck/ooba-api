import re
import sys
import json
import time
import torch
import threading
import transformers
from pathlib import Path
from modules import shared
from modules.models import load_model
from modules.LoRA import add_lora_to_model
from modules.models import clear_torch_cache
from modules.text_generation import encode, generate_reply, stop_everything_event

import uvicorn
from typing import Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
#from sse_starlette.sse import EventSourceResponse

from fastapi.responses import StreamingResponse
import asyncio
import io

# Tracking global variables here instead of shared:
task_id = 0
current_task = -1
pending_tasks = {}
finished_tasks = []

def start_new_thread(callback_function):
    new_thread = threading.Thread(target=callback_function)
    new_thread.start()

def _threaded_queue_callback():
    global pending_tasks
    print("callback")
    check_queue()
    

def check_queue():
    global pending_tasks
    print(pending_tasks)
    
    if len(pending_tasks)>=1 and current_task==-1:
        next_job = next(iter(pending_tasks))
        start_task(next_job)
        return 1
    else:
        return 0

def search_dict(dict, key):
    try:
        position = list(dict.keys()).index(int(key))
        return position+1
    except ValueError:
        pass

def start_task(id_task):
    global pending_tasks
    global current_task
    global time_start

    time_start = time.time()
    current_task = id_task
    req = pending_tasks.pop(id_task, None)
    print("start job: {0}".format(id_task))

    # generate request
    generate(req)


def finish_task():
    global current_task
    global finished_tasks

    finished_tasks.append(current_task)
    print("task finished: {0}".format(current_task))
    current_task = -1

    # check queue for any more tasks:
    check_queue()

    if len(finished_tasks) > 16:
        finished_tasks.pop(0)

def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

# Setup FastAPI:
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    #position in queue
    progress: str = Field(default=None, title="Progress", description="The number of messages in the queue")
    eta: float = Field(default=None, title="ETA in secs")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")


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

def add_task_to_queue(req: GenerateRequest):
    global task_id
    global pending_tasks

    task_id = task_id+1

    pending_tasks[task_id] = req
    #pending_tasks[task_id] = time.time()

    return task_id

@app.get("/")
def hellow_world(q: Union[str, None] = None):
    return {"wintermute": "ai", "q": q}


# call this instead of generate to join the queue.
@app.post("/queue", response_model=ProgressResponse)
def queue_job(req: GenerateRequest):
    global current_task
    global pending_tasks
    global finished_tasks

    # add to queue:
    task_id = add_task_to_queue(req)
    print("queue task: {0}".format(task_id))

    position = search_dict(pending_tasks, task_id)
    active = int(task_id) == int(current_task)
    queued = int(task_id) in pending_tasks
    completed = int(task_id) in finished_tasks


    # Queue has to recognize when we are the only person in queue and just start streaming, 
    # and don't callback.     This should be an else to that:

    # callback to handle pending tasks
    start_new_thread(_threaded_queue_callback)

    # Maybe instead of starting a thread, we should just wait await a generate() here?

    return ProgressResponse(active=active, queued=queued, completed=completed, progress="({0} out of {1})".format(position,len(pending_tasks)), textinfo="In queue..." if queued else "Waiting...")


@app.post("/internal/progress", response_model=ProgressResponse)
def progress(req: ProgressRequest):
    global current_task
    global pending_tasks
    global finished_tasks

    position = search_dict(pending_tasks, req.id_task)
    active = int(req.id_task) == int(current_task)
    queued = int(req.id_task) in pending_tasks
    completed = int(req.id_task) in finished_tasks
    
    if not active:
        return ProgressResponse(active=active, queued=queued, completed=completed, progress="(pos {0} out of {1})".format(position,len(pending_tasks)), textinfo="In queue..." if queued else "Waiting...")

    return ProgressResponse(active=active, queued=queued, completed=completed, progress="{0}".format(1), textinfo="currently processing")

# in generate strip to the last . rather than ending in the middle of a sentence. (?)
@app.post("/generate")
def generate(req: GenerateRequest):
    print(req.prompt)

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{0}
### Response:
""".format(req.prompt)

    prompt_lines = [k.strip() for k in prompt.split('\n')]
    prompt = '\n'.join(prompt_lines)

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

    # finish task in queue
    finish_task()

    print(answer.replace(prompt,""))
    return {"wintermute": "ai", "response": answer.replace(prompt,"")}


@app.post("/stream")
async def stream_data(req: GenerateRequest):
    print(req.prompt)

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{0}
### Response:
""".format(req.prompt)

    prompt_lines = [k.strip() for k in prompt.split('\n')]
    prompt = '\n'.join(prompt_lines)

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
    
    # start generating response:
    generator = generate_reply(
        prompt,
        generate_params,
        eos_token=None,
        stopping_strings=[],
    )

    async def gen():

        # If in queue and not processing, start a different stream?
        # yield from queue. /queue should return stream
        # ...

        _len = 0
        answer = ""
        answer_str = ""
        last_answer = ""
        for a in generator:
            print("LA: {0}".format(_len), flush=True)
            
            if isinstance(a, str):
                answer = a
            else:
                answer = a[0]

            # remove prompt from response:
            answer = answer.replace(prompt,"")
            # remove last part of the stream from response:
            answer = answer[_len:]

            # set next last_answer
            last_answer = answer
            _len = len(last_answer)
            yield answer.encode("utf-8")

    """
    answer = ''
    last_answer = ''
    for a in generator:
        last_answer = answer

        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]

        print(answer, flush=True)
    """

    return StreamingResponse(gen())
    #return StreamingResponse(generator)


@app.post("/agenerate")
async def agenerate(req: GenerateRequest):
    print("stream.")

@app.get("/check")
def check():
    return "Ok"

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

    #python main.py --model_type llama --wbits 4 --groupsize 128 --xformers
    #python main.py --model_type llama --load-in-8bit --lora baize-lora-13B --xformers
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7861
    )