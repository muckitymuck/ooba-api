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

<<<<<<< HEAD
# BIG THANKS:
=======
Make sure to edit `.env.example` and set the appropriate CUDA version for your GPU, which can be found on [developer.nvidia.com](https://developer.nvidia.com/cuda-gpus).

You need to have docker compose v2.17 or higher installed in your system. For installation instructions, see [Docker compose installation](https://github.com/oobabooga/text-generation-webui/wiki/Docker-compose-installation).

Contributed by [@loeken](https://github.com/loeken) in [#633](https://github.com/oobabooga/text-generation-webui/pull/633)

### Updating the requirements

From time to time, the `requirements.txt` changes. To update, use this command:

>>>>>>> 47daf891feb08e794b0579cfd59665fc2dda4b6e
```
https://github.com/oobabooga/text-generation-webui
https://github.com/1b5d/llm-api
https://github.com/abetlen/llama-cpp-python/tree/main/llama_cpp
```
<<<<<<< HEAD
=======
## Downloading models

Models should be placed inside the `models` folder.

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some examples:

* [Pythia](https://huggingface.co/models?sort=downloads&search=eleutherai%2Fpythia+deduped)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)

You can automatically download a model from HF using the script `download-model.py`:

    python download-model.py organization/model

For example:

    python download-model.py facebook/opt-1.3b

If you want to download a model manually, note that all you need are the json, txt, and pytorch\*.bin (or model*.safetensors) files. The remaining files are not necessary.

#### GPT-4chan

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) has been shut down from Hugging Face, so you need to download it elsewhere. You have two options:

* Torrent: [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Direct download: [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

The 32-bit version is only relevant if you intend to run the model in CPU mode. Otherwise, you should use the 16-bit version.

After downloading the model, follow these steps:

1. Place the files under `models/gpt4chan_model_float16` or `models/gpt4chan_model`.
2. Place GPT-J 6B's config.json file in that same folder: [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. Download GPT-J 6B's tokenizer files (they will be automatically detected when you attempt to load GPT-4chan):

```
python download-model.py EleutherAI/gpt-j-6B --text-only
```

## Starting the web UI

    conda activate textgen
    cd text-generation-webui
    python server.py

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

#### Basic settings

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--notebook`                               | Launch the web UI in notebook mode, where the output is written to the same text box as the input. |
| `--chat`                                   | Launch the web UI in chat mode. |
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA`                              | Name of the LoRA to apply to the model by default. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--no-stream`                              | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this json file. See `settings-template.json` for an example. If you create a file called `settings.json`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow.|
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maxmimum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.|
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to `cache/`. |
| `--load-in-8bit`                            | Load the model with 8-bit precision.|
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost. |
| `--xformers`                                | Use xformer's memory efficient attention. This should increase your tokens/s. |
| `--sdp-attention`                           | Use torch 2.0's sdp attention. |

#### llama.cpp

| Flag        | Description |
|-------------|-------------|
| `--threads` | Number of threads to use in llama.cpp. |

#### GPTQ

| Flag                      | Description |
|---------------------------|-------------|
| `--wbits WBITS`           | GPTQ: Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported. |
| `--model_type MODEL_TYPE` | GPTQ: Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported. |
| `--groupsize GROUPSIZE`   | GPTQ: Group size. |
| `--pre_layer PRE_LAYER`   | GPTQ: The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. |

#### FlexGen

| Flag             | Description |
|------------------|-------------|
| `--flexgen`                       | Enable the use of FlexGen offloading. |
| `--percent PERCENT [PERCENT ...]` | FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0). |
| `--compress-weight`               | FlexGen: Whether to compress weight (default: False).|
| `--pin-weight [PIN_WEIGHT]`       | FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%). |

#### DeepSpeed

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--deepspeed`                         | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK`             | DeepSpeed: Optional argument for distributed setups. |

#### RWKV

| Flag                            | Description |
|---------------------------------|-------------|
| `--rwkv-strategy RWKV_STRATEGY` | RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8". |
| `--rwkv-cuda-on`                | RWKV: Compile the CUDA kernel for better performance. |

#### Gradio

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--listen`                            | Make the web UI reachable from your local network. |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3" |

Out of memory errors? [Check the low VRAM guide](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

## Presets

Inference settings presets can be created under `presets/` as text files. These files are detected automatically at startup.

By default, 10 presets by NovelAI and KoboldAI are included. These were selected out of a sample of 43 presets after applying a K-Means clustering algorithm and selecting the elements closest to the average of each cluster.

[Visualization](https://user-images.githubusercontent.com/112222186/228956352-1addbdb9-2456-465a-b51d-089f462cd385.png)

## System requirements

Check the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) for some examples of VRAM and RAM usage in both GPU and CPU mode.

## Contributing

Pull requests, suggestions, and issue reports are welcome. 

You are also welcome to review open pull requests.

Before reporting a bug, make sure that you have:

1. Created a conda environment and installed the dependencies exactly as in the *Installation* section above.
2. [Searched](https://github.com/oobabooga/text-generation-webui/issues) to see if an issue already exists for the issue you encountered.

## Credits

- Gradio dropdown menu refresh button, code for reloading the interface: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Verbose preset: Anonymous 4chan user.
- NovelAI and KoboldAI presets: https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings-Presets
- Code for early stopping in chat mode, code for some of the sliders: https://github.com/PygmalionAI/gradio-ui/
>>>>>>> 1d8526849bfa6f1801da70912a875ae111c1e5af
