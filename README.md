# Chunk Size Analysis
The idea behind this repo comes from [this](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5) LlamaIndex project. The main changes are:
- Using [Groq](https://groq.com/) Llama3.2 11b for QA and Llama3 70b for evaluation
- Creating another run focused on LangChain
- making 3 runs of the script, storing the outputs at `output.md` for both LlamaIndex and LangChain folders
- Removing deprecated content
- Asking the model to create test cases based on an User Story besides answering questions provided from the evaluation LLM
  
In order to run it:
1. Set your own `GROQ_API_KEY` environment variable. You can create your key [here](https://console.groq.com/keys)
2. Run the installation requirements command bellow
   ```bash
   pip install llama-index llama-index-embeddings-huggingface llama-index-llms-groq spacy
   ```
3. Run the main script for LlamaIndex.

   3.1. If you want to evaluate question answering from the `llama-index/data/context` files, use the command below:
   ```bash
   python llama-index/chunk_size_discovery.py -qa
   ```
   3.2. If you want to evaluate test case generation from the `llama-index/data/context` files, use the command below:
   ```bash
   python llama-index/chunk_size_discovery.py -tc
   ```
4. Run the main script for LangChain.

   4.1. If you want to evaluate question answering from the `lang-chain/data/context` files, use the command below:
   ```bash
   python lang-chain/chunk_size_discovery.py -qa
   ```
   4.2. If you want to evaluate test case generation from the `lang-chain/data/context` files, use the command below:
   ```bash
   python lang-chain/chunk_size_discovery.py -tc
   ```
