# script for setting up all relevant models, consult the slides for instalation instructions
# curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_MAX_LOADED_MODELS=2 # sets the max number of loaded models
export OLLAMA_NUM_PARALLEL=2 # sets the max number of parallel tasks
# ollama serve # if not already running
ollama pull llama3
ollama pull gemma2:9b
ollama pull nomic-embed-text