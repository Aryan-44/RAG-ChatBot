import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
import tempfile
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate
load_dotenv()

api_key = os.getenv("groq-api-key")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0.5
)

st.title("ARYAN GPT")
st.write("Place any type of data, and chat with RAG Chatbot and clarify your doubts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

url = st.text_input("Enter URL")

button = st.button("Submit")

main_placeholder = st.empty()

embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    )


if button:

    all_documents = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete = False,
            suffix = os.path.splitext(file.name)[1],
            )as tmp_file: 

            tmp_file.write(file.getvalue())
            temp_path = tmp_file.name

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                main_placeholder.text("Loading data...")
                data = loader.load()
                for doc in data:
                    doc.metadata["file_name"] = file.name
                all_documents.extend(data)

            elif file.name.endswith(".docx"):
                loader =  Docx2txtLoader(temp_path)
                main_placeholder.text("Loading data...")
                data = loader.load()
                for doc in data:
                    doc.metadata["file_name"] = file.name
                all_documents.extend(data)

            elif file.name.endswith(".txt"):
                loader =  TextLoader(temp_path)
                main_placeholder.text("Loading data...")
                data = loader.load()
                for doc in data:
                    doc.metadata["file_name"] = file.name
                all_documents.extend(data)

            elif file.name.endswith(".csv"):
                loader =  CSVLoader(temp_path)
                main_placeholder.text("Loading data...")
                data = loader.load()
                for doc in data:
                    doc.metadata["file_name"] = file.name
                all_documents.extend(data)

            elif file.name.endswith(".xlsx"):
                loader =  UnstructuredExcelLoader(temp_path)
                main_placeholder.text("Loading data...")
                data = loader.load()
                for doc in data:
                    doc.metadata["file_name"] = file.name
                all_documents.extend(data)

    if url:
        loader = UnstructuredURLLoader(urls = [url])
        main_placeholder.text("Loading data...")
        data = loader.load()
        for doc in data:
                    doc.metadata["file_name"] = file.name
        all_documents.extend(data)

    # st.write(len(all_documents))
    # st.write(all_documents[0].metadata)
    # st.write(all_documents[0].page_content[:200])

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","," "],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Splitting data into chunks...")
    docs = text_splitter.split_documents(all_documents)

    # st.write(f"Original Documents: {len(all_documents)}")
    # st.write(f"Chunks Created: {len(docs)}")
    # st.write(docs[0].page_content[:300])

    vector_store = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    vector_store.save_local("faiss_index")
    main_placeholder.text("Vector database saved successfully!")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask anything...")

if query:

    st.session_state.messages.append(
        {
            "role" : "user",
            "content" : query
        }
    )

    with st.chat_message("user"):
        st.write(query)

    chat_history = ""
    for msg in st.session_state.messages[-6:]:
        chat_history += f"{msg['role']}: {msg['content']}\n"

    if os.path.exists("faiss_index"):

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs_and_scores = vectorstore.similarity_search_with_score(
            query,
            k=5
        )

        # for doc, score in docs_and_scores:
        #     st.write(doc.metadata)

        # for doc, score in docs_and_scores:
        #     st.write(f"Score: {score}")
        #     st.write(doc.page_content[:200])

        context = "\n\n".join(
            [doc.page_content for doc, score in docs_and_scores]
        )

        # st.subheader("Retrieved Context")
        # st.write(context[:2000])

        prompt = PromptTemplate.from_template(
            """
            You are an advanced AI assistant capable of both:

            1. Answering questions from uploaded documents.
            2. Acting as a general-purpose AI assistant like ChatGPT.

            Guidelines:

            1. Carefully analyze the user's question before answering.
            2. If relevant information exists in the provided document context, use it as the primary source of truth.
            3. When answering from document context, prioritize the document information over general knowledge.
            4. If the document context only partially answers the question, supplement the answer with your general knowledge while clearly distinguishing additional information.
            5. If the document context does not contain relevant information, answer using your general knowledge.
            6. If no document context is provided, behave as a normal AI assistant and answer naturally.

            Document Handling Rules:

            7. When summarizing documents, provide a concise but comprehensive summary.
            8. When comparing multiple uploaded documents, clearly identify similarities, differences, and key insights.
            9. If information is derived from uploaded documents, mention it.
            10. Never contradict information explicitly present in the document context.
            11. If information is not present in the documents, state that and then provide general knowledge if appropriate.

            General Assistant Rules:

            12. Answer technical, programming, AI/ML, data science, mathematics, and educational questions accurately.
            13. Use simple explanations unless advanced detail is requested.
            14. For coding questions, provide practical solutions.
            15. Structure long answers using headings and bullet points.
            16. If the user's question is ambiguous, make a reasonable interpretation.

            Conversation History:
            {chat_history}

            Document Context:
            {context}

            User Question:
            {question}

            Answer:
            """
        )

        router_prompt = PromptTemplate.from_template(
            """
            You are a relevance checker.

            Your task is to determine whether the provided document context contains useful information for answering the user's question.

            Rules:
            1. Answer YES if the context contains relevant information.
            2. Answer YES if the context partially answers the question.
            3. Answer NO if the context is unrelated.
            4. Answer NO if the context does not help answer the question.
            5. Respond with ONLY one word:
               YES
               or
               NO

            Document Context:
            {context}

            User Question:
            {question}

            Answer:
            """
        )

        general_prompt = PromptTemplate.from_template(
        """
        Previous Conversation:
        {chat_history}

        User Question:
        {question}

        Answer:
        """
        )

        document_used = False

        document_keywords = [
        "uploaded",
        "document",
        "documents",
        "file",
        "files",
        "pdf",
        "csv",
        "docx",
        "xlsx",
        "spreadsheet",
        "summary",
        "summarize",
        "summarise",
        "analyze",
        "analyse",
        "explain this file",
        "explain this document",
        "compare documents"
    ]

        if any(keyword in query.lower() for keyword in document_keywords):
            document_used = True
            decision = "YES"

        else:
            router_chain = router_prompt | llm

            router_res = router_chain.invoke(
                {
                    "question": query,
                    "context": context
                }
            )

            decision = router_res.content.strip().upper()

        st.write("Router Decision:", decision)

        if decision == "YES":

            chain_extract = prompt | llm

            res = chain_extract.invoke(
                {
                    "question": query,
                    "context": context,
                    "chat_history" : chat_history
                }
            )

            st.session_state.messages.append(
                {
                    "role" : "assistant",
                    "content" : res.content
                }
            )

            with st.chat_message("assistant"):
                st.write(res.content)

            if document_used:

                st.subheader("Sources")

                seen = set()

                for doc, score in docs_and_scores:

                    file_name = doc.metadata.get(
                        "file_name",
                        "Unknown Document"
                    )

                    page = doc.metadata.get(
                        "page",
                        None
                    )

                    if page is not None:
                        source_text = f"📄 {file_name} | Page {page + 1}"
                    else:
                        source_text = f"📄 {file_name}"

                    if source_text not in seen:
                        seen.add(source_text)
                        st.write(source_text)

        else:

            st.write("Using General Chat Mode")

            general_chain = general_prompt | llm

            res = general_chain.invoke(
                {
                    "question" : query,
                    "chat_history" : chat_history
                }
            )

            st.session_state.messages.append(
                {
                    "role" : "assistant",
                    "content" : res.content
                }
            )

            with st.chat_message("assistant"):
                st.write(res.content)

            




    

    
    

