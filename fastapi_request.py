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
    print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")

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


def generate(message):
    data = {
        "prompt": message,
        "temperature": 0.7,
        "log": True
        #"streaming": False
    }

    r = requests.post("http://wintermute:7861/generate", data=json.dumps(data), stream=True)

    if r.status_code==200:
        for chunk in r.iter_content(chunk_size=64):
            if chunk:
                print(chunk.decode("utf-8"), end="", flush=True)

    # append chunks in for loop..
    # then get the number of tokens.. return from generate:
    #tokens = encode(body['prompt'])[0]
    # that way we can use it in the timing function to get tokens/sec

#---

# Here we generate the test script:
# we can accept tuple: (models, loras) if we want to do lora later?
# want to accept questions[] list, or file.
def test_model(models):
    print("[test_model]:")

    if not isinstance(models, list):
        models = [models]

    for model in models:
        set_model(model)

        print(f"Testing model {model}:")
        gen_time = time_function_execution( generate, "question 1" )
        gen_time = time_function_execution( generate, "question 2" )
        gen_time = time_function_execution( generate, "question 3" )
        # put model, question, answer, and time all into a mysql database
        print()

    print("End of testing.")


#-------

if __name__ == "__main__":
    # Generate response:
    #gen_time = time_function_execution( generate, sys.argv[1] )

    # Get Number of Tokens:
    print( get_tokens("How many tokens is this?") )

    # Get, Set modals:
    #print( get_models() )
    #print( set_model("vicuna-13B-1.1-4bit") )

    # Get, Set text:
    #print( get_loras() )
    #print( set_loras(["homer"]) )

    # Test model:
    #test_model(["alpaca-30b-lora-4bit-128g"])
    #test_model("vicuna-13B-1.1-4bit")
    #test_model(["llamaOG-13B-hf", "koala-13B-HF"])

    # test with / without:
    # --quant-attn --warmup-autotune, --fused_mlp