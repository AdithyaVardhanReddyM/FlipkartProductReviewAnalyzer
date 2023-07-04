import streamlit as st
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import pinecone


load_dotenv()

def extracter(link):
    html_data_list = []
    
    link = link.replace("/p/", "/product-reviews/")
   
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    div_elements = soup.find_all("div", class_="t-ZTKy")
    
    for div_element in div_elements:
        inner_div = div_element.find("div", class_="")
        if inner_div:
            html_data = inner_div.get_text().strip() 
            html_data_list.append(html_data)

    # for multiple pages product review extraction
    
    # {base_link = link.replace("page=1","page=")
    
    # for i in range(2,6):
        
    #     link = base_link + str(i)
    #     response = requests.get(link)
    #     soup = BeautifulSoup(response.content, "html.parser")
    #     div_elements = soup.find_all("div", class_="t-ZTKy")
        
    #     for div_element in div_elements:
    #         inner_div = div_element.find("div", class_="")
    #         if inner_div:
    #             html_data = inner_div.get_text().strip() 
    #             html_data_list.append(html_data)

        html_data_list_filtered = []
    for html_data in html_data_list:
        filtered_html_data = html_data.replace('READ MORE', '')
        html_data_list_filtered.append(filtered_html_data)

    extracted_text = "\n".join(html_data_list_filtered)
    
    return extracted_text


def main():
    st.set_page_config(page_title="Flipkart product review Analyzer")
    st.header("Flipkart product review Analyzer")
    link = st.text_input("Paste the product link here")
    text = ""
    if link:
        text = extracter(link)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        #embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") #instructor-xl performs better than OpenAI but very slow
        
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'at pinecone.io')
        PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'create an index to get') #vector dimensions = 768 for instructor-xl and 1536 for OpenAI
        
        
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_API_ENV  # next to api key in console
        )
        index_name = "langchaintest"
        
        docsearch = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
        
        query = input("Assume all these are product reviews so, summarize or analyze all these and give your review")
        docs = docsearch.similarity_search(query)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)
    
    
        # index_name = 'faiss'
        # vectorstore = FAISS.from_texts([t.content for t in chunks], embedding=embeddings, index_name=index_name)
        # query = input("Assume all these are product reviews so, summarize or analyze all these and give your review")
        # docs = vectorstore.similarity_search(query=query, k=3)
        #llm = OpenAI(temperature=0)
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        # chain = load_qa_chain(llm=llm, chain_type="stuff")
        # response = chain.run(input_documents=docs, question=query)
        # st.write(response)
    

if __name__ == '__main__':
    main()
