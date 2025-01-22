# Chunk Size Analysis
This Repo is a fork from [this](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5) LlamaIndex article. The main changes are:
- making 10 runs of the script, storing the outputs at `output.md`
- Using [Groq](https://groq.com/) Llama3.2 11b for QA and Llama3 70b for evaluation
- Removing deprecated content
  
To run it:
1. Set your own `GROQ_API_KEY` environment variable. You can create your key [here](https://console.groq.com/keys)
2. Run the installation requirements command bellow
   ```bash
   pip install llama-index llama-index-embeddings-huggingface llama-index-llms-groq spacy
   ```
3. Run the main script:
   ```bash
   python chunk_size_discovery.py
   ```

