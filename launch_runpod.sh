cd /workspace

#---[Clone Ooba]----

git clone https://github.com/oobabooga/text-generation-webui.git
cd /workspace/text-generation-webui
pip install -r requirements.txt

#---[Clone GPTQ]----

mkdir repositories
cd /workspace/text-generation-webui/repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd /workspace/text-generation-webui/repositories/GPTQ-for-LLaMa
pip install -r requirements.txt

#---[Patches]----

# ooba break, i fix:
pip install --force gradio==3.28.3

# fix 8 bit lora training/saving:
# pip install bitsandbytes==0.37.2

#---[Clone Model]----

cd /workspace/text-generation-webui/models
git clone https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit
cd /workspace/text-generation-webui/models/alpaca-lora-65B-GPTQ-4bit
rm *.safetensors
wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/tokenizer.model -O tokenizer.model
#wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/alpaca-lora-65B-GPTQ-4bit-128g.safetensors -O alpaca-lora-65B-GPTQ-4bit-128g.safetensors
# 1024g
wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/alpaca-lora-65B-GPTQ-4bit-1024g.safetensors -O alpaca-lora-65B-GPTQ-4bit-1024g.safetensors

#---[Start Server]----

cd /workspace/text-generation-webui/
#python server.py --share --warmup_autotune --quant_attn --fused_mlp --wbits 4 --groupsize 128
# 1024g
python server.py --share --warmup_autotune --quant_attn --fused_mlp --wbits 4 --groupsize 1024