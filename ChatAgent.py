import requests

class ChatAgent:
    def __init__(self, personality):
        self.personality = personality

    def generate_message(self, chat_history):
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

        # resolve chat_history, personality here
        # pass user_input in as _MESSAGE for API to resolve. (better db)
        payload = {
            "personality": self.personality,
            "chat_history": chat_history
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["generated_text"]
