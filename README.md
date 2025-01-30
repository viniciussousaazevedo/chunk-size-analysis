# Chunk Size Analysis
The idea behind this repo comes from [this](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5) LlamaIndex project. The main changes are:
- Using [Groq](https://groq.com/) Llama3.2 11b for Q&A and Llama3 70b for evaluation
- Testing LangChain TextSplitter encapsulated in LlamaIndex framework
- making 3 runs of the Q&A script, storing the statistics output at `output/qa_output.md`
- Removing deprecated content and questions datasets
- Asking the model to create test cases based on an User Story with different chunk sizes(Besides the Q&A statistics part), storing the model outputs at `output/ct_gen_output.md`
  
In order to run it:
1. Set your own `GROQ_API_KEY` environment variable. You can create your key [here](https://console.groq.com/keys)
2. Run the installation requirements command bellow
   ```bash
   pip install llama-index llama-index-embeddings-huggingface llama-index-llms-groq spacy langchain
   ```
3. (Optional) Set context for Q&A script by replacing the pdf file in `data/qa`
4. (Optional) Set context for Test Case Generation by replacing the pdf file in `data/ct_gen/context`
   
   4.1. If you do this, remember to also replace the JSON file in `data/ct_gen/us` with your own made up User Story. Remember to use the same fields names on it.
5. Run the main script
   
   5.1. If you want to evaluate question answering, use the command below:
   ```bash
   python main.py -qa
   ```
   5.2. If you want to generate test cases, use the command below:
   ```bash
   python main.py -tc
   ```
