import sys
import json
import time
import requests


# [ RENAME: FASTAPI_REQUEST.py ]


def time_function_execution(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")

    return elapsed_time


def clear_loras():
    r = requests.get("http://wintermute:7861/clear_loras")
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
        #"streaming": False
    }

    r = requests.post("http://wintermute:7861/generate", data=json.dumps(data), stream=True)

    if r.status_code==200:
        for chunk in r.iter_content(chunk_size=64):
            if chunk:
                print(chunk.decode("utf-8"), end="", flush=True)


#---

# Here we generate the test script:
# we can accept tuple: (models, loras) if we want to do lora later?
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

    # Get, Set modals:
    #print( get_models() )
    #print( set_model("llamaOG-13B-hf") )

    # Get, Set text:
    #print( get_loras() )
    #print( set_loras(["homer"]) )

    test_model(["llamaOG-13B-hf", "koala-13B-HF"])
