cd /workspace

#---[Clone Ooba]----

git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
pip install -r requirements.txt

#---[Clone GPTQ]----

mkdir repositories
cd repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt

#---[Patches]----

# ooba break, i fix:
pip install --force gradio==3.28.3

# fix 8 bit lora training/saving:
# pip install bitsandbytes==0.37.2

#---[Clone Model]----

cd ../../models
git clone https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit
cd alpaca-lora-65B-GPTQ-4bit
rm *.safetensors
wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/tokenizer.model -o tokenizer.model
wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/alpaca-lora-65B-GPTQ-4bit-128g.safetensors -o alpaca-lora-65B-GPTQ-4bit-128g.safetensors
# 1024g
#wget https://huggingface.co/TheBloke/alpaca-lora-65B-GPTQ-4bit/resolve/main/alpaca-lora-65B-GPTQ-4bit-1024g.safetensors -o alpaca-lora-65B-GPTQ-4bit-1024g.safetensors

#---[Start Server]----

cd ../../
python server.py --share --warmup_autotune --quant_attn --fused_mlp --wbits 4 --groupsize 128
# 1024g
#python server.py --share --warmup_autotune --quant_attn --fused_mlp --wbits 4 --groupsize 1024