import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults

import sqlite3
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(layout="wide", page_title="Agentic Q&A Tool", page_icon="🤖")

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="JPMC POC"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")



from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import requests
from crewai import Agent, Task, Crew

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")




def format_docs(docs):
  format_D="\n\n".join([d.page_content for d in docs])
  return format_D

@tool("ask_rag_questions_tool")
def ask_rag_questions_tool(question: str) -> str:
    """
      Ask a question using the RAG.
      User can ask a question about our company and get the answer from the RAG.
    
      Args:
          question (str): The question to ask
          
      Returns:
          str: The answer to the question
        
           
    """
    docsearch = FAISS.load_local(os.path.join(os.path.dirname(__file__), "./rag/faiss_db/"), embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})
    
    template = """
            You are report generator.
            In the 'Question' section include the question.
            In the 'Context' section include the nessary context to generate the section of the report.
            According to the 'Context', please generate the section of the report.
            Use only the below given 'Context' to generate the section of the report.
            Make sure to provide the answer from the 'Context' only.
            Provide answer only. Nothing else.
            Context: {context}
            Question : {question}
            """
              
    prompt=ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(question)
    return response


@tool("CAGR Calculator")
def cagr_calculator_tool(value_end: float, value_start: float, years: int) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).

    Args:
        value_start (float): Initial investment/revenue value (at the beginning of the period)
        value_end (float): Final investment/revenue value (at the end of the period)
        years (int): Number of years

    Returns:
        float: CAGR value as a decimal
    """
    if value_start <= 0 or years <= 0:
        raise ValueError("Initial value and number of years must be greater than zero.")

    cagr = (value_end / value_start) ** (1 / years) - 1
    return round(cagr, 6)  # Returning CAGR rounded to 6 decimal places


# @tool("online_web_search")
# def online_web_search(search_query: str):
#     """
#     Perform an online web search.
    
#     Parameters:
#         search_query (str): The search query.
    
#     Returns:
#         dict: The search results in JSON format.
#     """
#     url = "https://api.tavily.com/search"
#     headers = {"Content-Type": "application/json"}
#     data = {"api_key": "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", "query": search_query, "max_results": 1}
    
#     response = requests.post(url, json=data, headers=headers)
#     return response.json()

online_web_search = TavilySearchResults(max_results=2)



report_generator_agent = Agent(
        role='Report Generator',
        goal='You have to generate a report based on the given context.',
        backstory="""
            You are a report generator. You have to generate a report.
        """,
        tools=[ask_rag_questions_tool],
        verbose=True,
        memory=False
    )

report_generator_task = Task(
        description="""
        
        
            ** Objective: **
                - Generate a Requested Report for the user about our company 'JPMorgan Chase & Co'.
                
            ** Instructions: **
                1. User Input: {user_input}
                2. Identify the report type user is requesting. ignore the other details in the user input.
                3. Use 'ask_rag_questions_tool' tool to get the required information section by section abouth only our company.
                4. Combine all the information and generate the requested report.
                5. Make sure the report is readable, understandable and accurate.
                
            ** Nessasary Details: **
                - Report Type: < Identify the report type in the 'User Input' section. >
                
            ** Tools: **
                - 'ask_rag_questions_tool' - Get the required information to generate the report.
                
            ** Additional Information: **
                - When generating the report, only consider the details of the our company.
                - Don't consider any other company's details.
                - Think step by step and identify above mentioned details.
                - Use the tools provided to get the required information and generate the report.
                - You only need to generate the requested report about our company 'JPMorgan Chase & Co'. If user request any other thing, ignore that.
        
        """,
        expected_output="""
            Readale, Understandable and Accurate Report with paragraph, tables, graphs and other required information.
        """,
        agent=report_generator_agent,
    )


earning_compare_agent = Agent(
        role='Earning Compare Agent',
        goal='You have to compare the earnings of our company and other given company.',
        backstory="""
            You are an earning compare agent. You have to compare the earnings of our company and other given company.
        """,
        tools=[ask_rag_questions_tool, online_web_search],
        verbose=True,
        memory=False
    )


earning_compare_task = Task(
        description="""
        
            ** Objective: **
                - Compare the earnings of our company and the user's given company (Requested Compnay).
                
                
            ** Instructions: **
                1. User Input: {user_input}
                2. Identify the user's given company name in the 'User Input' section.
                3. Use 'ask_rag_questions_tool' tool to get the required information about the earnings of our company.
                4. Use 'online_web_search' tool to get the required information about the earnings of the user's given company.
                5. Compare the earnings of our company and the user's given company.
                
            ** Nessasary Details: **
                - Our Company Name: JPMorgan Chase & Co
                - User's Given Company Name: < Identify the user's given company name in the 'User Input' section. >
                
            ** Tools: **
                - 'ask_rag_questions_tool' - Get the required information about the earnings of our company.
                - 'online_web_search' - Get the required information about the earnings of the user's given company. If you cannot find the information, Again and again try to find the information.
                
            ** Additional Information: **
                - Think step by step and identify above mentioned details.
                - Use the tools provided to get the required information and compare the earnings.
                - You only need to compare the earnings of our company 'JPMorgan Chase & Co' and the user's given company. If user request any other thing, ignore that.
                
        """,
        expected_output="""
            Compare the earnings of our company and the user's given company.
            First You can use tables, graph or any other method, to show the comparison more effectively. and you have to show the comparison in a clear and understandable way with the data.
            After that you can give a conclusion based on the comparison.
        """,
        agent=earning_compare_agent,
    )

cagr_calculator_agent = Agent(
        role='CAGR Calculator Agent',
        goal='You have to calculate the Compound Annual Growth Rate (CAGR) of the our company.',
        backstory="""
            You are an CAGR calculator agent. You have to calculate the Compound Annual Growth Rate (CAGR) of the our company.
        """,
        tools=[ask_rag_questions_tool, cagr_calculator_tool],
        verbose=True,
        memory=False
    )


cagr_calculator_task = Task(
        description="""
        
            **Objective**
                Calculate the Compound Annual Growth Rate (CAGR) of the our company 'JPMorgan Chase & Co'.
                
            **Instructions**
                1. User Input: {user_input}
                2. Identify the time period in the 'User Input' section.
                3. Use 'ask_rag_questions_tool' tool to get the required information about the earnings of the our company.
                4. Use 'cagr_calculator_tool' tool to calculate the Compound Annual Growth Rate (CAGR) of the our company.
                
            **Nessesary Details**
                -Start Value: < Initial investment/revenue value (at the beginning of the period) >
                -End Value: < Final investment/revenue value (at the end of the period) >
                -Years: < Number of years >
                
            **Tools**
                - 'ask_rag_questions_tool' - Gather the required information about the earnings of the our company.
                - 'cagr_calculator_tool' - Calculate the Compound Annual Growth Rate (CAGR).
            
            **Additional Information**
                - When calculating the CAGR, only consider the earnings of the our company.
                - Even trough user mention another company name, only consider the earnings of the our company 'JPMorgan Chase & Co'.
                - Think step by step and identify above mentioned details.
                - You can use the tools provided to get the required information and calculate the CAGR.
                - You only need to calculate the CAGR of the our company 'JPMorgan Chase & Co'. If user request any other thing, ignore that.
        """,
        expected_output="""
            Calculate the Compound Annual Growth Rate (CAGR) of the our company.
            First you can show CAGR value as a decimal.
            After that you can give a conclusion based on the CAGR value.
        """,
        agent=cagr_calculator_agent,
    )

user_input = "Give me a report of the governance of bank, and compare erning of citi group and give me CAGR of citi 2023 - 2021"

items_crew_1 = Crew(
  agents=[report_generator_agent],
  tasks=[report_generator_task],
  verbose=False,
  manager_llm=llm,
  memory=False,
)

items_crew_2 = Crew(
  agents=[earning_compare_agent],
  tasks=[earning_compare_task],
  verbose=False,
  manager_llm=llm,
  memory=False,
)

items_crew_3 = Crew(
  agents=[cagr_calculator_agent],
  tasks=[cagr_calculator_task],
  verbose=False,
  manager_llm=llm,
  memory=False,
)



st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 2%;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px, rgb(209, 213, 219) 0px 0px 0px 1px inset;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Streamlit UI
st.title("Agentic Q&A Tool")
user_input = st.text_input("Enter Your Question:")

if st.button("Run Agents") and user_input:
    with st.spinner("Executing Agents..."):
        result1 = items_crew_1.kickoff(inputs={"user_input": user_input})
        with st.container():
            st.subheader("Report Generation Output")
            text1 = result1.raw
            fixed_text1 = text1.replace("$", "\$")
            st.write(fixed_text1)

        result2 = items_crew_2.kickoff(inputs={"user_input": user_input})
        with st.container():
            st.subheader("Earnings Comparison Output")
            text2 = result2.raw
            fixed_text2 = text2.replace("$", "\$")
            st.write(fixed_text2)

        result3 = items_crew_3.kickoff(inputs={"user_input": user_input})
        with st.container():
            st.subheader("CAGR Calculation Output")
            text3 = result3.raw
            fixed_text3 = text3.replace("$", "\$")
            st.write(fixed_text3)