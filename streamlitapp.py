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
from src.mcqGen.mcqGenrator import genrate_evaluate_chain

import streamlit as st


with open('D:\GenAI\Projects\mcqGenrator\Response.json','r') as f:
    RESPONSE_JSON = json.load(f)



st.title("MCQs Creater Application with Langchain")

with st.form("User Input"):
    upload_file = st.file_uploader("Upload a PDF ot txt file")

    mcq_count = st.number_input("No. of MCQs",max_value=50,min_value=3)

    subject = st.text_input("Insert Subject",max_chars=20)

    tone = st.text_input("Complexity level of question",max_chars=20,placeholder="simple")

    buttom = st.form_submit_button("Create MCQs")

    if buttom and upload_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(upload_file)

                response = genrate_evaluate_chain(
                        { "text":text,
                            "numbers":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                            }
                            )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")

            else:
                if isinstance(response,dict):

                    quiz = response.get("quiz",None)

                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1

                            st.table(df)

                        else:
                            st.error("Error in the table data")

                    else:
                        st.write(response)




