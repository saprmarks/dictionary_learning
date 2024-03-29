#!/bin/bash

# Base URL for downloads - replace this with your actual URL, excluding the /aX part
BASE_URL="https://baulab.us/u/smarks/autoencoders/pythia-70m-deduped"

# Local directory where you want to replicate the folder structure, now including the specified root directory
LOCAL_DIR="dictionaires/pythia-70m-deduped"

# Default 'a' values array
declare -a default_a_values=("attn_out_layerX" "mlp_out_layerX" "resid_out_layerX") # Removed "embed" from default handling

# 'c' values array - Initially without "checkpoints"
declare -a c_values=("ae.pt" "config.json")

# Name of the set of autoencoders
sae_set_name="10_32768"

# Checkpoints flag variable, default is 0 (don't download checkpoints)
download_checkpoints=0

# Custom layers to download
declare -a custom_layers=("0" "1" "2" "3" "4" "5" "embed")

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--checkpoints) download_checkpoints=1 ;;
        --layers) IFS=',' read -ra custom_layers <<< "$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure the root download directory exists
mkdir -p "${LOCAL_DIR}"

# Include checkpoints if flag is set
if [[ $download_checkpoints -eq 1 ]]; then
    c_values+=("checkpoints")
fi

# Prepare 'a' values based on custom_layers input
declare -a a_values=()
if [[ ${#custom_layers[@]} -eq 0 ]]; then
    a_values=("${default_a_values[@]}")
else
    for layer in "${custom_layers[@]}"; do
        if [[ $layer == "embed" ]]; then
            a_values+=("embed")
        else
            for a_value in "${default_a_values[@]}"; do
                a_values+=("${a_value/X/$layer}")
            done
        fi
    done
fi

# Download logic
for a_value in "${a_values[@]}"; do
    for c in "${c_values[@]}"; do
        DOWNLOAD_URL="${BASE_URL}/${a_value}/${sae_set_name}/${c}"
        LOCAL_PATH="${LOCAL_DIR}/${a_value}/${c}"
        if [ "${c}" == "checkpoints" ]; then
            # Special handling for downloading checkpoints as folders
            mkdir -p "${LOCAL_PATH}"
            wget -r -np -nH --cut-dirs=7 -P "${LOCAL_PATH}" --accept "*.pt" "${DOWNLOAD_URL}/"

        else
            # Handle all other files
            mkdir -p "$(dirname "${LOCAL_PATH}")"
            wget -P "$(dirname "${LOCAL_PATH}")" "${DOWNLOAD_URL}"
        fi
    done
done

echo "Download completed."
