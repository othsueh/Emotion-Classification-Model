from huggingface_hub import snapshot_download
import tomllib
from utils import *

models = [
    # {"model": "whisper-large-v3", "source": "openai/whisper-large-v3"},
    # {"model": "wav2vec2-large-960h", "source": "facebook/wav2vec2-large-960h"},
    {"model": "wav2vec2-base-960h", "source": "facebook/wav2vec2-base-960h"},
    # {"model": "hubert-xlarge-ls960-ft", "source": "facebook/hubert-xlarge-ls960-ft"},
    # {"model": "wavlm-large", "source": "microsoft/wavlm-large"},
    # {"model": "llama-2-7b", "source": "meta-llama/Llama-2-7b-hf"},
    # {"model": "roberta-large", "source": "FacebookAI/roberta-large"},
    # {"model": "wav2vec2-large-960h-lv60-self", "source": "facebook/wav2vec2-large-960h-lv60-self"},
    # {"model": "t5-base", "source": "google-t5/t5-base"},
    # {"model": "qwen2.5-7b", "source": "Qwen/Qwen2.5-7B"},
    ]

if __name__ == "__main__":
    output_dir = config["PATH_TO_PRETRAINED_MODELS"]
    for m in models:
        new_dir = os.path.join(output_dir, m["model"])
        print(f'Checking {new_dir}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        print('=' * 30 + f'\nDownloading {m["model"]}\n' + '=' * 30)
        snapshot_download(m["source"], local_dir=new_dir, token=HUGGINGFACE_TOKEN)

