#!/bin/sh

# Downloads Voxtral Q4 GGUF model + tokenizer from HuggingFace.
# Usage: ./scripts/download-voxtral-model.sh [models_path]

BOLD="\033[1m"
RESET='\033[0m'

# Default to `models/` relative to the script root (one level up)
script_path="$(cd -- "$(dirname "$0")/.." >/dev/null 2>&1 && pwd -P)"
models_path="${1:-$script_path/models}"

mkdir -p "$models_path"

# ---- GGUF model (~2.5 GB) ----
gguf_url="https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf/resolve/main/voxtral-q4.gguf"
gguf_dst="$models_path/voxtral-q4.gguf"

if [ -f "$gguf_dst" ]; then
    printf "Model %s already exists. Skipping.\n" "$gguf_dst"
else
    printf "Downloading voxtral-q4.gguf (2.5 GB) ...\n"
    if [ -x "$(command -v wget2)" ]; then
        wget2 --no-config --progress bar -O "$gguf_dst" "$gguf_url"
    elif [ -x "$(command -v curl)" ]; then
        curl -L --output "$gguf_dst" "$gguf_url"
    elif [ -x "$(command -v wget)" ]; then
        wget --no-config --quiet --show-progress -O "$gguf_dst" "$gguf_url"
    else
        printf "Either wget2, curl, or wget is required.\n"
        exit 1
    fi
    if [ $? -ne 0 ]; then
        printf "Failed to download voxtral-q4.gguf\n"
        exit 1
    fi
fi

# ---- Tokenizer (~15 MB) ----
tok_url="https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf/resolve/main/tekken.json"
tok_dst="$models_path/tekken.json"

if [ -f "$tok_dst" ]; then
    printf "Tokenizer %s already exists. Skipping.\n" "$tok_dst"
else
    printf "Downloading tekken.json (15 MB) ...\n"
    if [ -x "$(command -v wget2)" ]; then
        wget2 --no-config --progress bar -O "$tok_dst" "$tok_url"
    elif [ -x "$(command -v curl)" ]; then
        curl -L --output "$tok_dst" "$tok_url"
    elif [ -x "$(command -v wget)" ]; then
        wget --no-config --quiet --show-progress -O "$tok_dst" "$tok_url"
    else
        printf "Either wget2, curl, or wget is required.\n"
        exit 1
    fi
    if [ $? -ne 0 ]; then
        printf "Failed to download tekken.json\n"
        exit 1
    fi
fi

printf "\n${BOLD}Done!${RESET} Model files saved in:\n"
ls -lh "$gguf_dst" "$tok_dst"

printf "\n${BOLD}Usage:${RESET}\n"
printf "  voxtral_load_model(ctx, \"%s\", \"%s\");\n" "$gguf_dst" "$tok_dst"
