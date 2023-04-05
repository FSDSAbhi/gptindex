from gpt_index import SimpleDirectoryReader,GPTListIndex,GPTSimpleVectorIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os
import openai

def answerMe(vectorindex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorindex)
    while True:
        question = input("Please ask:")
        response = vIndex.query(question,response_mode = "compact")
        print(f"Response: {response} \n")

answerMe('saved_vector_index.json') 