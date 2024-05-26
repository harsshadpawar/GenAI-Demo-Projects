from IPython import display
import streamlit as st
import os
#import openai
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from IPython.display import Markdown
load_dotenv()

def RAGLogic(response):
    import os
    import chromadb
    from llama_index.llms.openai import OpenAI
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext, get_response_synthesizer, Settings, PromptTemplate
    from llama_index.readers.file import PDFReader, CSVReader
    from IPython.display import display, Markdown
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLAMA_INDEX_API_KEY = os.getenv("LLAMA_INDEX_API_KEY")  
    
    # Load in a specific embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", chunk_size=1024, chunk_overlap=200, embed_batch_size=32,api_key=OPENAI_API_KEY)

    # Load in a specific language model
    llm = OpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY,temperature=0.7, max_tokens=3000, top_p=1, logprobs=True, echo=True, stream=True)

    # Load global settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    # initialize client
    db = chromadb.PersistentClient(path="./chroma_db")

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
    )

    # configure retriever
    retriever = VectorIndexRetriever( index=index, similarity_top_k=5,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine( retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )


    # Create Custom Prompt Template
    template = (
    "You are NutriGuide, an AI assistant specializing in personalized nutrition advice. You provide diet plans and nutritional guidance based on ICMR guidelines and individual health reports. Start each response with 'NutriGuide:'.\n"
    "**Context Information**\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "**User Question:** {query_str}\n"
    )

    context_str = (
    "User Health Report Summary:\n"
    "- Condition: High blood pressure\n"
    "- Age: 34\n"
    "- Gender: Female\n"
    "- Weight: 74 kg\n"
    "- Height: 160 cm\n"
    "- Lifestyle: Sedentary\n"
    "- Specific dietary restrictions: None\n"
    )

    qa_template = PromptTemplate(template)


    # create a query engine and query
    from IPython.display import Markdown

    query_engine = index.as_query_engine(text_qa_template=qa_template)
    return query_engine.query(response)


    #display(Markdown(f"<b>{response}</b>"))

with st.sidebar:
    # Load your image (replace with your image path)
    
    loaded_image = Image.open("./images/AI-Nutritionist.jpg") 
    st.sidebar.image(loaded_image)  # Adjust width as needed

    # Sidebar content
    st.sidebar.title("💬MyAssistant 🦙:")

    st.sidebar.header("Trained on:")
    st.sidebar.write("💳 ICMR Guidelines")
    st.sidebar.write("🧾 Clinical Reports")
    st.sidebar.write("🛒 Diet Plan")
   

    loaded_image4 = Image.open("./images/LLama-Index-Logo.png")
    st.sidebar.image(loaded_image4) 
    loaded_image2 = Image.open("./images/openai-logo.png")
    st.sidebar.image(loaded_image2)

    #st.sidebar.image(image)  # Place the image within the sidebar
    openai_api_key = os.getenv("OPENAI_API_KEY")  


st.header("Gen AI in Nutrition!(Demo)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    #st.chat_message(msg["role"]).write(msg["content"])
    formatted_content = f"{msg['content']}"
    st.chat_message(msg["role"]).write(formatted_content)
if prompt := st.chat_input("Ask me anything!"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    st.chat_message("user").write(prompt)
    
    # Use your query engine to generate a response
    response = RAGLogic(prompt)
    
   # msg = response.choices[0].message.content
    
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(f"{response}")
