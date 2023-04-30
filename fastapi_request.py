import sys
import json
import time
import requests


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
        _PROMPT = "" #message

    # setup json payload:
    data = {
        "prompt": _PROMPT,
        "message": message, 
        "temperature": 0.7, # set to 1 for evals for reproducability?
        "log": True,
        #"streaming": False
    }

    r = requests.post("http://wintermute:7861/generate", data=json.dumps(data), stream=True)

    if r.status_code==200:
        for chunk in r.iter_content(chunk_size=64):
            if chunk:
                print(chunk.decode("utf-8"), end="", flush=True)


# chatgpt endpoint for convience.. it uses my website's endpoint because i dont want to expost my open ai api key :P you are free to re-implement with your own, instead!
def chatgpt(message):
    # currently only supporting 1 message at a time, not a chain of messages, as that is not what I am using this for.
    data = {
        "messages": [{"role": "user", "content": message}], 
        "temperature": 0.7, # set to 1 for evals for reproducability?
    }

    r = requests.post("https://3jane.net/generate", data=json.dumps(data), stream=True)
    
    if r.status_code == 200:
        _RESPONSE = ""
        for line in r.iter_lines():
            # Filter out keep-alive new lines and decode the content
            if line:
                decoded_line = line.decode('utf-8')

                # Remove the "data: " prefix
                json_line = decoded_line.replace("data: ", "")

                try:
                    # Load the JSON data into a Python dictionary
                    json_data = json.loads(json_line)

                    if 'chunk' in json_data:
                        # Extract the inner JSON strings from the 'chunk' key and split them by     newline characters
                        inner_json_strings = json_data['chunk'].replace("data: ", "").split('\n')

                        for inner_json_string in inner_json_strings:
                            if inner_json_string.strip():
                                # Load the inner JSON string into a Python dictionary
                                inner_json_data = json.loads(inner_json_string)

                                # Access the desired content
                                content = inner_json_data['choices'][0]['delta'].get('content', '')
                                _RESPONSE += content
                                print(content, end="", flush=True)
                    else:
                        # Directly access the desired content from the json_data
                        content = json_data['choices'][0]['delta']['content']
                        print(content)

                except json.JSONDecodeError as e:
                    pass
                    #print(f"Error decoding JSON: {e}")

        return { "response": _RESPONSE }
    else:
        return { "status_code": r.status_code}

            
# Here we generate the test script:
# we can accept tuple: (models, loras) if we want to do lora later?
def test_model(models):
    print("[test_model]:")

    if not isinstance(models, list):
        models = [models]

    #questions:
    _QUESTIONS = [
        "What is the best way to make money with a 100W laser cutter?",

        "There are 2 boys playing with 2 balls. One is red and One is blue. One of the boys is colorblind. What color do you think the balls are?",

        "Rewrite this CSS style='width:500px' in tailwind.css", 

        "Write me a python function that uses Requests to make a post request to this endpoint http://example.com/endpoint with the json payload 'message': 'booyah!'",

        "Anna takes a ball and puts it in a red box, then leaves the room. Bob takes the ball out of the red box and puts it into the yellow box, then leaves the room. Anna returns to the room. Where will she look for the ball",
    ] 

    for model in models:
        print(f"Loading model {model}:")
        set_model(model)


        for question in _QUESTIONS:
            generate(question)

    print("End of testing.")


#-------

if __name__ == "__main__":
    # Generate response:
    '''
    try:
        #generate(sys.argv[1])
        print( chatgpt(sys.argv[1]) )
    except Exception as e:
        print('Missing arguments, try: python3 fastapi_request.py "hello"')
        print("OR: {0}".format(str(e)))
    '''

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
    #test_model("koala-13B-HF-4bit")
    #test_model(["alpaca-30b-lora-4bit-128g"])
    #test_model(["vicuna-13B-1.1-4bit", "koala-13B-HF"])