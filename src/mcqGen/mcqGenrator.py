import os
import json
import pandas as pd
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
import PyPDF2
from dotenv import load_dotenv 
load_dotenv()

from src.mcqGen.utils import read_file,get_table_data
from src.mcqGen.logger import logging


import google.generativeai as genai
API_KEY = os.getenv("GEMNI_API_KEY")
genai.configure(api_key=API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=API_KEY,)






TEMPLATE = """
Test:{text}
You are an expert MCQ Maker. Given the above text, it is your job to 
create a quiz of {numbers} multiple choice question for {subject} students in {tone} tone.
Make sure the question are not repeated and check all the questions to be configured the text as well.
Make sure to format your response like response_json below and use it as a guide.
Ensure to make {numbers} mcqs
### response_json
{response_json}

"""


quiz_genration_prompt = PromptTemplate(
    input_variables=["text","numbers","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm,prompt=quiz_genration_prompt,output_key="quiz",verbose=True)


TEMPLATE2 = """
You are an expert english grammerian and writer . Given a multiple choice quiz for {subject} students.
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 word for complexity.
If the quiz is not as per with thw congnitive and analytical abilities of the student,
update the quiz question which need to be changed and cjange the tone such that it perfectly fit the student.
QUIZ_MCQ:
{quiz}

check for expert english writer of the above quiz
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject","quiz"],template=TEMPLATE2)


review_chain = LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

genrate_evaluate_chain = SequentialChain(chains=[quiz_chain,review_chain],
                      input_variables=["text","numbers","subject","tone","response_json"],
                      output_variables=["subject","quiz"],
                      verbose = True

                      
                      )


# file_path = "D:\GenAI\Projects\mcqGenrator\data.txt"
# with open(file_path,"r") as f:
#     TEXT = f.read()



# NUMBER = 5
# SUBJECT = "machine learning",
# TONE="simple"



# genrate_evaluate_chain(
#    { "text":TEXT,
#     "numbers":NUMBER,
#     "subject":SUBJECT,
#     "tone":TONE,
#     "response_json": json.dumps(response_json)
#     }
#     )