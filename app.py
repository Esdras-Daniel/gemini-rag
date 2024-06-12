from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from PyPDF2 import PdfReader

from dotenv import load_dotenv

import os
from io import StringIO

# Lê pares de chaves-valores de um arquivo .env e os adiciona como variáveis do ambiente.
load_dotenv()

def split_string_with_overlay(input, n_chars = 500, overlap = 0.1):
  # Lista para armazenar os pedaços resultantes
  result = []
  length = len(input)

  # Calcula o tamanho do overlap
  overlap = int(n_chars * overlap)

  # Ponto inicial de cada pedaço
  start = 0

  while start < length:
      end = start + n_chars
      piece = input[start:end]
      result.append(piece)

      # Atualiza o ponto inicial para o próximo pedaço
      start = start + n_chars - overlap

  return result

# Definindo os modelos de LLM e de Embeddings
model = ChatGoogleGenerativeAI(model='gemini-pro',
                               temperature=0.8)

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

output_parser = StrOutputParser()

# Configurando título da página e header
st.set_page_config(page_title='Gemini RAG')
st.header("Gemini RAG")

# Upload do arquivo .pdf
uploaded_file = st.file_uploader('Escolha o seu arquivo .pdf', type='pdf')

entry = st.text_input(label='Dúvida: ')
btn = st.button(label="Enviar")

pdf_content = ''
print(pdf_content)

if uploaded_file is not None and pdf_content == '':
    # Lendo o PDF e armazenando o conteúdo em pdf_content
    reader = PdfReader(uploaded_file)

    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        pdf_content += page.extract_text()
    
    print(len(pdf_content))

    # Criando a lista de pedaços do texto
    #pdf_content_split = split_string_with_overlay(pdf_content)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 50)
    pdf_content_split = splitter.split_text(pdf_content)

    print(len(pdf_content_split))

    # Criando o VectorStore
    vectorstore = DocArrayInMemorySearch.from_texts(pdf_content_split,
                                                    embedding=embeddings).as_retriever(k = 4)

# Verificando se o botão foi pressionado
if btn:
    # Criando template
    prompt = ChatPromptTemplate.from_template("""
                                              Responda a pergunda baseado apenas no seguinte contexto: {context}
                                              Pergunta: {question}
                                              """)
    
    print(vectorstore.get_relevant_documents(entry)[0])

    chain = RunnableMap({
        'context': lambda x: vectorstore.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    }) | prompt | model | output_parser

    ans = chain.invoke({'question': entry})

    st.write(ans)