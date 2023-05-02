import requests

#OpenAIgent (sensei bot)
#KoalaAgent / VicunaAgent (could be the same, class just running on different models.)

class ChatAgent:
    def __init__(self, personality):
        self.personality = personality

    def generate_message(self, chat_history, message):
        _PROMPT="""### Chat history: 
{chat_history}
 
### Personality:
{personality}

Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a response to the User message that uses elements of your personality!
### User message:
{user_input}
### Response:
"""

        # resolve chat_history, personality here:
        _PROMPT = _PROMPT.replace("{personality}", self.personality)
        _PROMPT = _PROMPT.replace("{chat_history}", chat_history)
        #print(_PROMPT)
        
        # pass user_input in as _MESSAGE for API to resolve. (better db)
        data = {
            "prompt": _PROMPT,
            "message": message, 
            "temperature": 0.7,
            "log": True,
            "evl": 0 #eval
        }

        r = requests.post("http://wintermute:7861/generate", json=data, stream=True)

        _RESPONSE = ""
        if r.status_code==200:
            for chunk in r.iter_content(chunk_size=64):
                if chunk:
                    print(chunk.decode("utf-8"), end="", flush=True)
                    _RESPONSE += chunk.decode("utf-8")


        return { "response": _RESPONSE } 

if __name__ == "__main__":
    # set up agents:
    agent_smith = ChatAgent("You are a lazy AI. You don't like to do much but you DO like talking about rockets.")
    agent_jane = ChatAgent("You are a chat agent with the personality of an ambitious cyber girl who loves to program and fix software bugs and hates pizza.")

    # start chat:
    _chat_history = ""
    print("\n[Smith]:\n")
    _msg = agent_smith.generate_message(_chat_history, "Hello, how are you?")
    _chat_history += _msg['response']

    # chat chain:
    #I think we can condense chat history and msg
    # maybe we can start by finding a prompt to let us add msg to end of chat
    while True:
        print("\n[Jane]:\n")
        _msg = agent_jane.generate_message(_chat_history, _msg['response'])
        _chat_history += _msg['response']
        print("\n[Smith]:\n")
        _msg = agent_smith.generate_message(_chat_history, _msg['response'])
        _chat_history += _msg['response']