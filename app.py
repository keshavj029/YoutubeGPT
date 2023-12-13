import os
from APIKey import apikey
 
import streamlit as st
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
#SimpleSequentialChain- it will output only the last given prompt in chain but in order to get outputs to multiple inputs we need sequentialchain

from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['GOOGLE_API_KEY'] = apikey

st.title("🦜🔗 Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write a Youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write a Youtube video script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

#Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history', return_messages=True)
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history', return_messages=True)

#LLms
llm = GooglePalm(temperature = 0.8)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key='script', memory = script_memory)
#sequential_chain = SimpleSequentialChain(chains = [title_chain, script_chain], verbose = True)
# sequential_chain = SequentialChain(chains = [title_chain, script_chain],input_variables= ['topic'], output_variables= ['title', 'script'], verbose = True)

wiki = WikipediaAPIWrapper()

#show stuff to the screen

if prompt:

    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research )
    # response = sequential_chain({'topic': prompt})
    #response = SimpleSequentialChain.run(prompt)
    #st.write(response)
    # st.write(response['title'])
    # st.write(response['script'])

    st.write(title)
    st.write(script)

    # with st.expander('Title History'):
    #     st.info(title_memory.buffer)
    
    # with st.expander('Script History'):
    #     st.info(script_memory.buffer)
    
    # with st.expander('Wikipedia History'):
    #     st.info(wiki_research)
    



