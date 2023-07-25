### Imports
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain
from langchain import PromptTemplate, LLMChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
import os
import openai
import io
from contextlib import redirect_stdout
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from datetime import datetime
import langchain
# from langchain.cache import InMemoryCache
# langchain.llm_cache = InMemoryCache()
# from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path="langchain.db")

### CSS
st.set_page_config(
    page_title='GAMEPLAN', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
padding_top = 0
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# OpenAI Credentials
if not os.environ["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]

### UI
""
col1, col2 = st.columns( [1,5] )
col1.image('628858.png', width=170)
col2.title('GamePlan')
col2.subheader('Generative AI Managed Enterprise PLAtform Network')
# st.markdown('---')

def change_q(myquestion):
    promt = myquestion

def run_cyber(myquestion):
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)

    ### Bring in my controlling documents and the additonal template
    with open('content/cybersecurity.txt') as f:
        tasks = f.readlines()
    mytasks = str(tasks)    
    template=f"""You are a cybersecurity expert. 
    Provide a detailed answer to the "QUESTION" below. 
    Use the "TEXT" below to help develop to steps necessary to answer this question.
    (When running a step the Thought/Action/Action Input/Observation can be repeated just 1 time.)
    If you can't answer the question return the answer: "Sorry, but I can't help you with that task."

    TEXT:
    {mytasks}

    QUESTION:
    {myquestion}
    Provide your justification for this answer.
    """
    
    ### Build an agent that can be used to run SQL queries against the database
    llm4 = ChatOpenAI(model="gpt-4", temperature=0, verbose=False)
    llm = ChatOpenAI(model=llm_model, temperature=0, verbose=False)
    mydb = SQLDatabase.from_uri("sqlite:///chinook.sqlite")
    toolkit = SQLDatabaseToolkit(db=mydb, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm, #OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=False
    )

    ### Build an agent that can perform mathematics...not used but provided as an example.
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    ### Build a chain from the three tools
    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="SQL",
            func=sql_agent.run,
            description="""This tool queries a SQLite database. It is useful for when you need to answer questions 
            by running SQLite queries. Always indicate if your response is a "thought" or a "final answer". 
            The following table information is provided to help you write your sql statement. 
            Use the emails table to determine email information.
            If you are looking for employee information within the emails table use the from_name or to_name column.
            Use the apache_logs table to determine access information.
            If you are looking for employee information within the apache_logs table use the "user" column.
            If you are looking for date or time information within the apache_logs table use the dt column. 
            
            apache_logs: (ID, ip, user, dt, tz, vrb, uri, resp, byt, referer, useragent)
            emails: (send_date, from_name, to_name, subject, body, attachment_type, filesize, sentiment)
            """
        )
    ]

    planner = load_chat_planner(llm)
    executor = load_agent_executor(
        llm4, 
        tools, 
        verbose=True,
    )
    pe_agent = PlanAndExecute(
        planner=planner, 
        executor=executor,  
        verbose=True, 
        max_iterations=2,
        # max_execution_time=180,
    )

    if show_detail:
        f = io.StringIO()
        with redirect_stdout(f):
            with st.spinner("Processing..."):
                response = pe_agent(template)
    else:
        with st.spinner("Processing..."):
            response = pe_agent(template)

    st.session_state.messages.append({"role": "assistant", "content": response['output']})    
    st.chat_message('assistant').write(response['output'])

    if show_detail:
        with st.expander('Details', expanded=False):
            s = f.getvalue()
            st.write(s)

with st.sidebar: 
    mysidebar = st.selectbox('Select GamePlan', ['Cybersecurity', 'Data Science'])
    if mysidebar == 'Cybersecurity':
        show_detail = st.checkbox('Show Details')
        llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
        st.markdown("---")
        st.markdown("### Standard Questions:")
        fit = st.button('Find Threats')
        offhours = st.button('Offhour Access')
    if mysidebar == 'Data Science':
        st.markdown("---")
        st.markdown("### Planner Chain:")
        st.markdown("&nbsp;&nbsp;&nbsp; Internet Search")
        st.markdown("### Executor Chain:")
        st.markdown("&nbsp;&nbsp;&nbsp; Python")
        st.markdown("&nbsp;&nbsp;&nbsp; SQL")
        st.markdown("&nbsp;&nbsp;&nbsp; Pandas")
        st.markdown("&nbsp;&nbsp;&nbsp; Memory")

if mysidebar == 'Cybersecurity':
    with st.expander("**:blue[Cybersecurity Overvew]**"):
        st.markdown("**:blue[The cybersecurity model uses a Planner/Executor SuperChain.]**")
        st.markdown("**:blue[The workflow first enters the Planner phase where it uses vector semantic KNN search of our CSIO's cybersecurity document to determine how to answer the question. This document resembles an FAQ document and provides steps for completing these tasks. These are tasks a human might take. The LLM will translate these steps into steps that LangChain can execute.]**")
        st.markdown("**:blue[In the Executor phase the LLM instructs LangChain how to perform each step. Answers from each step are preserved and a final answer is generated by the LLM to the proposed questions. For this demo LLM largely relies on SQL queries. However, the workflow could include other operations including dynamic Python using SciKit, querying the internet, running shell scripts, running REST queries, or any act that might be defined in the CSIO's document.]**")

        col1, col2, col3 = st.columns([15, 70, 15])
        col2.image('cyber.jpg',caption='LangChain Structure')

    st.markdown('---')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask a cybersecurity question?"):
        start = datetime.now()
        st.write("Start: "+str(start))
        run_cyber(prompt)
        st.write("End: "+str(datetime.now()))
        st.write("Duration: "+str(datetime.now() - start))
    if fit:
        start = datetime.now()
        st.write("Start: "+str(start))
        run_cyber("Who are our insider threats?")
        st.write("End: "+str(datetime.now()))
        st.write("Duration: "+str(datetime.now() - start))
    if offhours:
        start = datetime.now()
        st.write("Start: "+str(start))
        run_cyber("Using the apache_logs table, list the users and their total accesses that occur between the hours of 8pm and 6am.") 
        st.write("End: "+str(datetime.now()))
        st.write("Duration: "+str(datetime.now() - start))


if mysidebar == 'Data Science':

    agent_executor = create_python_agent(
        llm=OpenAI(temperature=0, max_tokens=1000),
        tool=PythonREPLTool(),
        verbose=True
    )

    st.markdown("### **:blue[Overview:]** ")
    st.markdown("**:blue[This gameplan is used to answer questions where dynamic Python is required. It also has the ability to query databases to extract the data necessary for the analysis.]**")
    st.markdown("**:blue[It can write regular python, SciKit, Pytorch...for example, ask *'write a single neuron neural network in PyTorch. Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs. Return prediction for x = 5'*]**")
    myquestion = st.text_input('Question', placeholder='Ask a question?', value="What is the 10th fibonacci number?")

    if myquestion:
        with st.spinner('Running LangChain...Please Wait...'):
            myresults = agent_executor.run(myquestion)
            st.write(myresults)

    st.markdown('---')
    with st.expander("See explanation"):
        st.markdown("**:blue[A Planner/Executor SuperChain is used in this model. In the Planner phase, the LLM searches the internet to determine how to solve the problem. It then developes a set of steps for the Executor phase.]**")
        st.markdown("**:blue[In the Executor phase the LLM instructs LangChain how to perform each step. The chain has access to the data lakehouse, dynamic Python, a Pandas tool, and a memory tool so data can be moved between tools.]**")

        col1, col2, col3 = st.columns([15, 70, 15])
        col2.image('python.jpg',caption='LangChain Structure')
