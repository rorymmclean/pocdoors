### Imports
import streamlit as st
import os
import io
import pymysql
from datetime import datetime
import pandas as pd

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain.db")
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool
from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_python_agent
from contextlib import redirect_stdout

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
col1.write("")
col1.image('GTLogo.png', width=170)
col2.write("")
col2.subheader('GT AI Driven Cyber Analytics Platform')
# col2.subheader('Generative AI Managed Enterprise PLAtform Network')
# st.markdown('---')

def change_q(myquestion):
    promt = myquestion

def run_cyber(myquestion):
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)

    ### Bring in my controlling documents and the additonal template
    relevancy_cutoff = .5
    myprompts = pd.read_csv('content/myprompts.csv')

    db3 = Chroma(persist_directory="./chromadb", 
                collection_name="runbook", 
                embedding_function=OpenAIEmbeddings())

    docs = db3.similarity_search_with_relevance_scores(myquestion, k=8)
    mytasks = ""
    for x,v in docs:
        if mytasks == "":
            mytasks = str(x)[14:-13] + "\n\n"
            continue
        if v < relevancy_cutoff:
            mytasks = mytasks + str(x)[14:-13] + "\n\n"

    template=f"""{myprompts[myprompts['name']=='Template']['prompt'].values[0]}
    TEXT:
    {mytasks}

    QUESTION:
    {myquestion}
    Provide your justification for this answer.
    """
    
    ### Build an agent that can be used to run SQL queries against the database
    llm4 = ChatOpenAI(model="gpt-4", temperature=0, verbose=False)
    llm = ChatOpenAI(model=llm_model, temperature=0, verbose=False)
    mydb = SQLDatabase.from_uri("mariadb+pymysql://streamlit:streamlitpass@localhost/streamlit?charset=utf8mb4")
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
            description=myprompts[myprompts['name']=='Calculator']['prompt'].values[0]
        ),
        Tool(
            name="SQL",
            func=sql_agent.run,
            description=myprompts[myprompts['name']=='SQL']['prompt'].values[0]
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
    mysidebar = st.selectbox('Select GamePlan', ['Cybersecurity'])
    if mysidebar == 'Cybersecurity':
        show_detail = st.checkbox('Show Details')
        llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
        st.markdown("---")
        st.markdown("### Standard Questions:")
        fit = st.button('Find Threats')
        offhours = st.button('Offhour Access')
        st.markdown("---")
        tz = st.container()

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
        tz.write("Start: "+str(start)[10:])
        run_cyber(prompt)
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if fit:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_cyber("Who are our insider threats?")
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if offhours:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        run_cyber("Using the \"apache_logs\" table, create and ordered list of the top five users based upon their total accesses that occur between the hours of 8pm and 6am. Sort the list in descending order by \"total accesses\". ") 
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
