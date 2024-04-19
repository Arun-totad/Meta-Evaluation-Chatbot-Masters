# META EVALUATION CHATBOT

## Large Language Processing (LLM): 
A language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification.

## LangChain: 
A framework designed to simplify the creation of applications using large language models. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.

## Ollama: 
A tool that allows you to run open-source large language models (LLMs) locally on your machine.

## ChainLit: 
An open-source Python package to build production ready Conversational AI.

## Chat Generative Pre-Trained Transformer for PDF meta-evaluation
Chatbot to process pdf documents using LLMs under Ollama(llama2) model locally, with LangChain and Chainlit.

- Prompt user to upload a document (Type = pdf)
- Generate chroma vector embeddings from that pdf and store in the memory for processing
- A graphical user interface chatbot with the ability to display sources used to generate an answer based on local model llama2 LLM

## System Requirements
- Must have Python 3.9 or later installed. Earlier versions of python may not compile.
- Ollama model installed in local machine