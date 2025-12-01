import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import shutil
from html_template import css, bot_template, user_template
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# openai_key = os.getenv("OPENAI_API_KEY")
# ----------------- Simple auth config (POC) ----------------- #

USERS = {
    "auditor": {
        "password": "auditor123",
        "role": "Auditor",
    },
    "app_owner": {
        "password": "owner123",
        "role": "App Owner",
    },
    "func_head": {
        "password": "func123",
        "role": "Functional Head",
    },
    "vertical_head": {
        "password": "vert123",
        "role": "Vertical Head",
    },
    "cio": {
        "password": "cio123",
        "role": "CIO",
    },
}


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an audit and compliance assistant for a bank.\n"
        "Always answer in clear, professional **English** only.\n\n"
        "Use ONLY the information in the context below to answer the question. "
        "If the answer is not in the context, say that you do not know "
        "and suggest that the user refer to the official RBI/bank documentation.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer in English:"
    ),
)


SUMMARY_PROMPT = (
    "You are an audit and compliance assistant for a bank. "
    "Always answer in clear, professional English only.\n\n"
    "Based on the retrieved documents and evidence, write a concise summary "
    "for the application '{app_name}', covering:\n"
    "- Key backup and cyber security controls in place\n"
    "- Important audit findings and management actions\n"
    "- Any apparent gaps or vulnerabilities versus RBI expectations\n\n"
    "If information is missing for any part, state that it is not clearly "
    "available in the ingested documents."
)


# Making the vectorstore shared across all users
VECTORSTORE_PATH = "faiss_index"
def save_vectorstore(vectorstore):
    vectorstore.save_local(VECTORSTORE_PATH)

def load_vectorstore():
    # check if FAISS files exist
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


# ----------------- Helper for 'Make Ingested Docs' shared for all roles ----------------- #
DOCS_META_PATH = "ingested_docs.json"

def load_docs_meta():
    if os.path.exists(DOCS_META_PATH):
        with open(DOCS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_docs_meta(docs):
    with open(DOCS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


# ----------------- Helper for Comments & Signoff ----------------- #
COMMENTS_FILE = "comments_signoff.json"
def load_comments_data():
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_comments_data(data):
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
# ----------------- RAG helper functions (from old app) ----------------- #


def get_text(pdf_docs):
    """Extract raw text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_chunks(raw_txt):
    """Split raw text into overlapping chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_txt)
    return chunks


def get_vectorstore(text_chunks):
    """Create an in-memory FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conv_chain(vectorstore):
    """Create a ConversationalRetrievalChain over the given vector store,
    with an English-only prompt and chat history."""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return conv_chain



# ----------------- Session helpers ----------------- #


def init_session():
    """Ensure required session keys exist."""
    defaults = {
        "logged_in": False,
        "username": None,
        "role": None,
        "conversation": None,
        "chat_history": [],
        "ingested_docs": [],  # list of {file_name, app_name, doc_type, uploaded_by}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def do_logout():
    """Clear auth + app state and rerun."""
    for key in [
        "logged_in",
        "username",
        "role",
        "conversation",
        "chat_history",
        "ingested_docs",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


# ----------------- Auth / login UI ----------------- #


def show_login():
    st.title("Audit Evidence Copilot – POC")

    st.markdown("#### Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    

    if st.button("Login"):
        user = USERS.get(username)
    
        if not user or user["password"] != password:
            st.error("Invalid username or password")
            return

        # Successful login
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = user["role"]

        st.success(f"Welcome, {username} ({user['role']})")
        st.rerun()


# ----------------- Dashboard tabs ----------------- #


def upload_and_ingest_tab(role: str):
    st.subheader("Upload & Ingest")

    if role not in ["Auditor", "App Owner"]:
        st.info("You do not have permission to upload documents.")
        return

    app_name = st.selectbox(
    "Application",
    ["NetBank", "TradeFin", "CoreBank"],
    key="upload_app_select",
)
    doc_type = st.selectbox(
        "Document type",
        ["Policy", "Procedure", "Audit Report", "Logs", "Config"],
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if st.button("Upload & Ingest") and uploaded_files:
        with st.spinner("Processing documents..."):
            # 1. Extract text from PDFs
            raw_txt = get_text(uploaded_files)

            if not raw_txt.strip():
                st.error("No text could be extracted from the uploaded PDFs.")
                return

            # 2. Split into chunks
            text_chunks = get_chunks(raw_txt)

            # 3. Build vector store
            vectorstore = get_vectorstore(text_chunks)

            # 4. Create conversational retrieval chain
            st.session_state.conversation = get_conv_chain(vectorstore)
            save_vectorstore(vectorstore)

            # 5. Reset chat history for new corpus
            st.session_state.chat_history = []

            # 6. Track ingested docs in session for View tab
            for f in uploaded_files:
                st.session_state.ingested_docs.append(
                {
                    "File Name": f.name,
                    "Application": app_name,
                    "Type": doc_type,
                    "Uploaded By": st.session_state.username,
                }
            )
            existing = load_docs_meta()
            existing.extend(st.session_state.ingested_docs[-len(uploaded_files):])
            save_docs_meta(existing)

        st.success(
            f"Ingested {len(uploaded_files)} file(s) for application '{app_name}' "
            f"as type '{doc_type}'."
        )


def ask_question_tab(role: str):
    st.subheader("Ask a Question")

    # Load shared vectorstore
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.info("No documents have been ingested yet. Please ask an App Owner or Auditor to upload evidence.")
        return

    if st.session_state.conversation is None:
        st.session_state.conversation = get_conv_chain(vectorstore)

    # Chat history display (reusing old HTML templates)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if hasattr(msg, "type"):
            msg_role = msg.type  # newer versions
        elif hasattr(msg, "role"):
            msg_role = msg.role  # older versions
        else:
            msg_role = "assistant"

        template = user_template if msg_role == "human" else bot_template
        st.write(
            template.replace("{{MSG}}", msg.content),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Input at bottom with on_change callback
    def submit_query():
        user_q = st.session_state.pending_query
        if st.session_state.conversation and user_q:
            response = st.session_state.conversation({"question": user_q})
            st.session_state.chat_history = response["chat_history"]
            st.session_state.pending_query = ""  # Clear after processing

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.text_input(
        "Type your message…",
        key="pending_query",
        on_change=submit_query,
    )


def reset_index_tab(role: str):
    st.subheader("Reset Index")

    if role not in ["Auditor", "App Owner"]:
        st.info("Only Auditor or App Owner can reset the index in this POC.")
        return

    if st.button("Reset All Ingested Data"):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.ingested_docs = []

        # delete shared FAISS index + docs metadata
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
        if os.path.exists(DOCS_META_PATH):
            os.remove(DOCS_META_PATH)

        st.success("Index reset. All ingested data cleared for this demo.")


def view_ingested_docs_tab(role: str):
    st.subheader("View Ingested Documents")

    docs = load_docs_meta()
    if not docs:
        st.info("No documents have been ingested yet.")
        return

    st.table(docs)


def backup_validation_evidence_tab(role: str):
    st.subheader("Backup Validation Evidence")

    st.write(
        "Create or review 'evidence packs' for specific controls "
        "(e.g. RBI backup requirements)."
    )
    st.info("Placeholder – wire this to question/answer + selected source docs later.")


def comments_signoff_tab(role: str):
    st.subheader("Comments & Signoff")

    username = st.session_state.username
    comments_data = load_comments_data()

    # 1) Select application
    app_name = st.selectbox(
        "Application",
        ["NetBank", "TradeFin", "CoreBank"],
        key="cs_app_select",
    )

    # Filter items for this app
    app_items = [i for i in comments_data if i["app_name"] == app_name]

    # 2) Search + table of topics
    st.markdown("#### Review topics for this application")

    if app_items:
        table_rows = []
        for i in app_items:
            s = i["signoffs"]
            table_rows.append(
                {
                    "ID": i["id"],
                    "Topic": i["topic"],
                    "Created At": i["created_at"].replace("T", " ").replace("Z", ""),
                    "App Owner": s.get("App Owner", "Pending"),
                    "Functional Head": s.get("Functional Head", "Pending"),
                    "Vertical Head": s.get("Vertical Head", "Pending"),
                    "CIO": s.get("CIO", "Pending"),
                }
            )
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No review items yet for this application (or none match the search).")

    # 3) Select an existing topic to view/edit
    if app_items:
        selected_id = st.selectbox(
            "Select a topic to view / edit",
            options=[i["id"] for i in app_items],
            format_func=lambda x: next(i["topic"] for i in app_items if i["id"] == x),
            key="cs_select_topic",
        )
        st.session_state.current_cs_item_id = selected_id
    else:
        selected_id = None
        st.session_state.current_cs_item_id = None

    # 4) Create a NEW topic explicitly
    st.markdown("#### Create new topic / control")
    new_topic = st.text_input(
        "New Topic / Control ID",
        key="cs_new_topic",
        placeholder="e.g. Backup Policy Compliance",
    )
    if st.button("Create new review item"):
        if not new_topic.strip():
            st.warning("Topic cannot be empty.")
        else:
            # avoid duplicate for same app + topic
            exists = any(
                i["app_name"] == app_name and i["topic"] == new_topic
                for i in comments_data
            )
            if exists:
                st.warning("A review item with this topic already exists for this app.")
            else:
                new_id = max([i["id"] for i in comments_data], default=0) + 1
                new_item = {
                    "id": new_id,
                    "app_name": app_name,
                    "topic": new_topic,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "comments": [],
                    "signoffs": {
                        "App Owner": "Pending",
                        "Functional Head": "Pending",
                        "Vertical Head": "Pending",
                        "CIO": "Pending",
                    },
                }
                comments_data.append(new_item)
                save_comments_data(comments_data)
                st.success(f"Created new review item for '{new_topic}' in {app_name}.")
                st.session_state.current_cs_item_id = new_id
                st.rerun()

    # If nothing selected, stop here
    if not st.session_state.current_cs_item_id:
        return

    # 5) Detail view (comments + signoff) for selected topic
    item = next(
        (i for i in comments_data if i["id"] == st.session_state.current_cs_item_id),
        None,
    )
    if not item:
        st.error("Selected review item not found.")
        return

    st.markdown(
        f"**Current Item:** {item['topic']}  \n"
        f"_App:_ {item['app_name']}"
    )

    col1, col2 = st.columns(2)

    # --- Comments ---
    with col1:
        st.markdown("#### Comments")

        if item["comments"]:
            for c in item["comments"]:
                ts = c["created_at"].replace("T", " ").replace("Z", "")
                st.markdown(
                    f"- **{c['user']} ({c['role']})** at {ts}:  \n  {c['text']}"
                )
        else:
            st.write("No comments yet.")

        new_comment = st.text_area(
            "Add a comment",
            key="cs_new_comment",
            placeholder="Write your observation, concern, or note here...",
        )
        if st.button("Add Comment"):
            if not new_comment.strip():
                st.warning("Comment cannot be empty.")
            else:
                item["comments"].append(
                    {
                        "user": username,
                        "role": role,
                        "text": new_comment.strip(),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                )
                save_comments_data(comments_data)
                st.success("Comment added.")
                st.rerun()

    # --- Signoff ---
    with col2:
        st.markdown("#### Signoff Status")

        for r in ["App Owner", "Functional Head", "Vertical Head", "CIO"]:
            st.write(f"{r}: **{item['signoffs'].get(r, 'Pending')}**")

        signoff_roles = ["App Owner", "Functional Head", "Vertical Head", "CIO"]
        if role in signoff_roles:
            current_status = item["signoffs"].get(role, "Pending")
            new_status = st.selectbox(
                "Update your signoff status",
                ["Pending", "Approved", "Rework"],
                index=["Pending", "Approved", "Rework"].index(current_status)
                if current_status in ["Pending", "Approved", "Rework"]
                else 0,
                key="cs_signoff_status",
            )
            if st.button("Save Signoff"):
                item["signoffs"][role] = new_status
                save_comments_data(comments_data)
                st.success(f"Signoff updated to '{new_status}'.")
                st.rerun()
        else:
            st.info("Your role can comment but not change signoff status for this item.")




def generate_summary_tab(role: str):
    st.subheader("Generate Summary")

    # Use shared FAISS index directly (stateless)
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.info("Please upload and ingest documents first.")
        return

    app_name = st.selectbox(
        "Application",
        ["NetBank", "TradeFin", "CoreBank"],
        key="summary_app_select",
    )

    if st.button("Generate Summary for selected app"):
        with st.spinner("Generating summary from ingested documents..."):
            llm = ChatOpenAI()

            # Simple RetrievalQA chain, no chat memory
            retriever = vectorstore.as_retriever()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
            )

            prompt = SUMMARY_PROMPT.format(app_name=app_name)
            summary = qa_chain.run(prompt)

        if summary:
            st.markdown("#### Summary")
            st.write(summary)
        else:
            st.warning("No summary could be generated.")




def show_dashboard():
    username = st.session_state.username
    role = st.session_state.role

    # Header
    st.markdown(
        f"### Audit Evidence Dashboard  \n"
        f"Logged in as: **{username}**  \n"
        f"Role: **{role}**"
    )

    if st.button("Logout"):
        do_logout()

    # Tabs (main navigation)
    tabs = st.tabs(
        [
            "Upload & Ingest",
            "Ask a Question",
            "Reset Index",
            "View Ingested Docs",
            "Backup Validation Evidence",
            "Comments & Signoff",
            "Generate Summary",
        ]
    )

    with tabs[0]:
        upload_and_ingest_tab(role)

    with tabs[1]:
        ask_question_tab(role)

    with tabs[2]:
        reset_index_tab(role)

    with tabs[3]:
        view_ingested_docs_tab(role)

    with tabs[4]:
        backup_validation_evidence_tab(role)

    with tabs[5]:
        comments_signoff_tab(role)

    with tabs[6]:
        generate_summary_tab(role)


# ----------------- Main entrypoint ----------------- #


def main():
    load_dotenv()
    st.set_page_config(page_title="Audit Evidence Copilot", layout="wide")

    init_session()

    # Global CSS for chat bubbles
    st.write(css, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        show_login()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
