# langchain-Ollama-Chainlit
Chat with documents using LLMs with Ollama (mistral model) locally, LangChaiin and Chainlit
  
- Upload a document(pdf)
- Create vector embeddings from that pdf
- Create a chatbot app with the ability to display sources used to generate an answer
---

## System Requirements
You must have Python 3.9 or later installed. Earlier versions of python may not compile.  
---

## Steps to Replicate 

1. Create a virtualenv and activate it
   ```
   python3 -m venv .venv
   .venv\Scripts\activate.bat
   
   ```
2. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the following command in the terminal to execute Chainlit
   ```
   chainlit run rag.py
   ```

   ---
