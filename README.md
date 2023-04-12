# Drop-in FastAPI wrapper for (oobabooga/text-generation-webui):
If you already have text-generation-webui set up you can just copy the `main.py` from this repo into your folder and you should be good to go!

> pip install fastapi
> python main.py --wbits 4  --groupsize 128 --model_type llama --xformers

You can call it with the same args as `server.py` from ooba.

# Purpose:
I want to write a FastAPI wrapper for Llama server running on GPU (without Gradio)

# Install:
```
conda create -n llm-api python=3.10.9
conda activate llm-api

conda install cudatoolkit
pip install torch torchvision torchaudio
```

**#  Clone this repo (disarmyouwitha/llm-api-gpu):**
```
git clone https://github.com/disarmyouwitha/llm-api
cd llm-api
pip install -r requirements.txt
```

**# (qwopqwop200/GPTQ-for-LLaMa):**
```
mkdir repositories
cd repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt
```
**# start server:**
```
python server.py --model_type llama --wbits 4 --groupsize 128 
```

# BIG THANKS:
```
https://github.com/oobabooga/text-generation-webui
https://github.com/1b5d/llm-api
https://github.com/abetlen/llama-cpp-python/tree/main/llama_cpp
```
