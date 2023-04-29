import sys
import json
import time
import requests


# modify generate to return tokens and then we can get tokens/sec
def time_function_execution(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\n[Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.]")

    return elapsed_time


def get_tokens(prompt):
    data = {
        "prompt": prompt,
    }

    r = requests.post("http://wintermute:7861/tokens",  json=data)
    return r.json()


def get_loras():
    r = requests.get("http://wintermute:7861/loras")
    return r.json()


def set_loras(loras):
    data = {
        "lora_names": loras,
    }

    r = requests.post("http://wintermute:7861/loras",  json=data)
    return r.json()


def get_models():
    r = requests.get("http://wintermute:7861/models")
    return r.json()


def set_model(model_name):
    data = {
        "model": model_name,
    }

    r = requests.post("http://wintermute:7861/models",  json=data)
    return r.json()


def generate(message, format="instruct"):
    #set prompt, change prompt: maybe move this into it's own fucntion:
    if "instruct" in format.lower():
        _PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{user_input}
### Response:"""
    elif "complete" in format.lower():
        _PROMPT = "{user_input} "
    elif "chat" in format.lower():
        _PROMPT = """User: {user_input}
AI:"""
    else:
        print("instruct format mismatch")
        _PROMPT = message

    # place user message into prompt: 
    _PROMPT = _PROMPT.replace("{user_input}", message)

    # setup json payload:
    data = {
        "prompt": _PROMPT,
        "temperature": 0.7, # set to 1 for evals for reproducability?
        "log": True,
        #"streaming": False
    }

    r = requests.post("http://wintermute:7861/generate", data=json.dumps(data), stream=True)

    if r.status_code==200:
        for chunk in r.iter_content(chunk_size=64):
            if chunk:
                print(chunk.decode("utf-8"), end="", flush=True)


# Here we generate the test script:
# we can accept tuple: (models, loras) if we want to do lora later?
# want to accept questions[] list, or file.
def test_model(models):
    print("[test_model]:")

    if not isinstance(models, list):
        models = [models]

    for model in models:
        print(f"Loading model {model}:")
        set_model(model)

        generate("What is the best way to make money with a 100W laser cutter?")
        generate("There are 2 boys playing with 2 balls. One is red and One is blue. One of the boys is colorblind. What color do you think the balls are?")
        generate("Rewrite this CSS style='width:500px' in tailwind.css")
        generate("Write me a python function that uses Requests to make a post request to this endpoint http://example.com/endpoint with the json payload 'message': 'booyah!'")
        generate("Anna takes a ball and puts it in a red box, then leaves the room. Bob takes the ball out of the red box and puts it into the yellow box, then leaves the room. Anna returns to the room. Where will she look for the ball?")
        #generate("")
        #generate("")

    print("End of testing.")


#-------

if __name__ == "__main__":
    # Generate response:
    #try:
    #    gen_time = time_function_execution( generate, sys.argv[1] )
    #except:
    #    print('Missing arguments, try: python3 fastapi_request.py "hello"')

    # Get Number of Tokens:
    #print( get_tokens("How many tokens is this?") )

    # Get, Set modals:
    #print( get_models() )
    #print( set_model("vicuna-13B-1.1-4bit") )

    # Get, Set text:
    #print( get_loras() )
    #print( set_loras(["homer"]) )

    # Test model:
    test_model("vicuna-13B-1.1-4bit")
    #test_model(["alpaca-30b-lora-4bit-128g"])
    #test_model(["vicuna-13B-1.1-4bit", "koala-13B-HF"])