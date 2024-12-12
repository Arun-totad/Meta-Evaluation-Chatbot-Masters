# Meta-Evaluation Chatbot

## Introduction
This project aims to create a chatbot that performs meta-evaluation, which involves evaluating human evaluations. The current scope of the project is to perform evaluations based on a set of hardcoded parameters. In the future, this can be extended to cater to various industries by optimizing the parameter values.

## Minimum Viable Product (MVP)
The MVP involves creating a chatbot that can:
- Accept a PDF document as input.
- Be context-aware and perform Natural Language Processing (NLP).
- Understand and summarize the text from the uploaded document and return an output.

## Methodology
The project utilizes the following technologies and methodologies:
- **LangChain**: An application development framework for Large Language Models (LLMs). It provides a set of abstractions and tools to build applications with LLMs as a core component.
- **Ollama (Llama2 model)**: An open-source language model based on the Llama2 architecture. It is a powerful and flexible model for various natural language processing tasks.
- **ChainLit**: An open-source Python platform for deploying and running LLM-based applications. It provides tools and utilities to simplify building and deploying LLM-powered applications.
- **Retrieval Augmented Generation (RAG)**: An approach used to feed custom data filtering features to a chatbot. It combines the power of large language models with the ability to retrieve and incorporate relevant information from external sources.

## RAG Architecture
RAG combines the power of large language models (LLMs) with the ability to retrieve and incorporate relevant information from external sources. This allows the chatbot to provide more accurate and contextual responses by leveraging both the broad knowledge of the LLM and the specific information retrieved from custom data sources.

## Evaluation Criteria
The evaluation process involves two phases:

### Phase I – Criteria
There are five criteria and 30 meta-evaluation standards used to score any primary evaluation:
1. Utility Standards
2. Feasibility Standards
3. Propriety Standards
4. Accuracy Standards
5. Evaluation Accountability Criteria

### Phase II – Design
The focus is mainly on Accuracy, which has eight meta-evaluation standards. Each standard has around six questions, reframed in a way that the response is True or False. True gets a score of 1, and False gets a score of 0. Each criterion is summed up, and a mathematical formula determines the final score.

## Challenges
- The huge size of PDF documents took a long time to process since we were executing locally.
- Some criteria are vague and have subjective answers, which may change when questions are reframed.
- Evaluation standards are very specific and useful only for these documents.

## References
- LangChain Documentation
- ChainLit Documentation
- LangChain Chatbots
- LangChain Vectorstores
