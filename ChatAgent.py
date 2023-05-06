import requests

# work on nodes/look at supercharger to see how he does load balancing.
# I want to run 2 13b agents on 1 card.. figure it out soon it is only 7 days until we have 2 cards!

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
Write a response to the Chat history!
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
    agent_josh = ChatAgent("You are an intergalactic rock star.")
    agent_jon = ChatAgent("""name: "Chiharu Yamada"
context: "Chiharu Yamada's Persona: Chiharu Yamada is a young, computer engineer-nerd with a knack for problem solving and a passion for technology."
greeting: |-
  *Chiharu strides into the room with a smile, her eyes lighting up when she sees you. She's wearing a light blue t-shirt and jeans, her laptop bag slung over one shoulder. She takes a seat next to you, her enthusiasm palpable in the air*
  Hey! I'm so excited to finally meet you. I've heard so many great things about you and I'm eager to pick your brain about computers. I'm sure you have a wealth of knowledge that I can learn from. *She grins, eyes twinkling with excitement* Let's get started!
example_dialogue: |-
  User: So how did you get into computer engineering?
  Chiharu: I've always loved tinkering with technology since I was a kid.
  User: That's really impressive!
  Chiharu: *She chuckles bashfully* Thanks!
  User: So what do you do when you're not working on computers?
  Chiharu: I love exploring, going out with friends, watching movies, and playing video games.
  User: What's your favorite type of computer hardware to work with?
  Chiharu: Motherboards, they're like puzzles and the backbone of any system.
  User: That sounds great!
  Chiharu: Yeah, it's really fun. I'm lucky to be able to do this as a job.""")

    # start chat:
    _chat_history = ""
    print("\n[Jon]: ", end="")
    _msg = agent_jon.generate_message(_chat_history, "Josh: What is your favorite animal and why?")
    _chat_history += f"Jon: {_msg['response']}\n"

    # chat chain:
    #I think we can condense chat history and msg
    # maybe we can start by finding a prompt to let us add msg to end of chat

    # When we break out of while loop it will print chat history.. that would be nice.
    while True:
        
        print("\n[Josh]: ", end="")
        _msg = agent_josh.generate_message(_chat_history, _msg['response'])
        _chat_history += f"Josh: {_msg['response']}\n"

        print("\n[Jon]: ", end="")
        _msg = agent_jon.generate_message(_chat_history, _msg['response'])
        _chat_history += f"Jon: {_msg['response']}\n"