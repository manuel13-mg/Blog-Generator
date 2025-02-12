import streamlit as st
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

## Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    ## LLama2 model
    llm = CTransformers(model='C:/Users/manua/blog generator/model/llama-2-7b.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

st.set_page_config(page_title="Blogs Generator",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Blogger 🤖")

input_text = st.text_input("Enter the Blog Topic")

## Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

## Final response
if submit:
    try:
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.markdown(response, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
