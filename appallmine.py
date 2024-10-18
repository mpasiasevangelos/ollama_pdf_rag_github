from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

#import torch
#cuda_available = torch.cuda.is_available()
#print(f"CUDA Available: {cuda_available}")
#if cuda_available:
#    print(f"Using CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Set the device to CUDA if available

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#deviceToUse=torch.cuda.current_device()
#print(torch.version.cuda)

# List of Ollama models to use
ollama_models = ["aya"]
                 #,"aya:35b", "wizardlm2:8x22b", "aya", "ymertzanis/meltemi7b", "llama3","phi3:latest" ]


files_path=["pan-bio-2.pdf"]


while True:
    for file in files_path:
      
        # Path to the PDF file
        local_path = file
    
        # Local PDF file uploads
        if local_path:
            loader = UnstructuredPDFLoader(file_path=local_path)
            data = loader.load()
        else:
            print("Upload a PDF file")
    
    # Preview first page
    #print(data[0].page_content)
    
    # Split and chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
    
        # Add to vector database
        vector_db = Chroma.from_documents(
            documents=chunks,
            #embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True, num_gpu=0),
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name="local-rag"
        )
        
        # Query prompt template
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        
        # RAG prompt
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Iterate over the models
        for model_name in ollama_models:
            print(model_name)
            # Initialize the LLM from Ollama
            llm = ChatOllama(model=model_name)
            #llm = ChatOllama(model=model_name, device='cuda')
            
            # Create the retriever with the multi-query retriever
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )
        
            # Define the chain for invoking the model
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
            # Invoke the chain with the user question
            question = "κάνε 10 ερωτήσεις με 4 απαντήσεις πολλαπλής επιλογής από το κείμενο που σου έδωσα και δώσε για κάθε μία ερώτηση την σωστή απάντηση, εννοείται στα Ελληνικά"
            response = chain.invoke({"question": question})
            # Save the response to a file
            with open(f"response_{model_name.replace(':', '_')}.txt", "a", encoding="utf-8") as file:
                file.write(f"Response from model {model_name}:\n")
                file.write(response)
                file.write("\n" + "="*50 + "\n")

            
            # Print the response from each model
            print(f"Response from model {model_name}:")
            print(response)
            print("\n" + "="*50 + "\n")