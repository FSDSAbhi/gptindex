'''
SimpleDirectoryReader: Used to read the PDF files/Knowledge Base
GPTListIndex: Indexing the data
GPTSimpleVectorIndex: Load the indexed data
LLMPredictor: Load the Large Language Model to be used
PromptHelper: Used to define the user prompts
'''
from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os
import openai

# define openai api key
openai.API_KEY = os.environ['OPENAI_API_KEY']

#filepath is the location of the files/documents
def createVectorIndex(filepath):
    max_input = 4096
    tokens = 256
    chunck_size = 600
    max_chunk_overlap = 50

    Prompt_helper = PromptHelper(max_input,tokens,max_chunk_overlap=max_chunk_overlap,chunk_size_limit=chunck_size)

    #define language model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0,
                                            top_p=0.5,
                                             model_name= 'text-ada-001',
                                             max_tokens=tokens))
    #load data
    docs = SimpleDirectoryReader(filepath).load_data()

    #Creating vector index
    vectorindex = GPTSimpleVectorIndex.from_documents(docs)
    vectorindex.save_to_disk('saved_vector_index.json')
    return vectorindex

vectorindex = createVectorIndex('/Users/abhilashmarecharla/Desktop/gptindex/documents/')
'''
(base) (env) abhi@MBA-FVFHH1LBQ6LX env % python main.py
INFO:gpt_index.token_counter.token_counter:> [build_index_from_nodes] Total LLM token usage: 0 tokens
INFO:gpt_index.token_counter.token_counter:> [build_index_from_nodes] Total embedding token usage: 114570 tokens
'''