import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever


# 1. Initialize RAG once
chunks = load_and_chunk_documents("./Data")
retriever, vectorstore = get_hybrid_retriever(chunks)
llm = ChatOllama(model="gemma4:e4b", temperature=0.0)

# 2. Tailored Prompts for each role
PATIENT_PROMPT = ChatPromptTemplate.from_template("""
Answer strictly using context:
{context}

Question: {question}
Answer in plain, easy-to-understand language for patients, include source citation and the relevance score percentage:""")

STAFF_PROMPT = ChatPromptTemplate.from_template("""
Answer strictly using context:
{context}

Question: {question}
Answer with full clinical technical details, source citation, and the relevance score percentage:""")

# 3. Context formatter with role-based filtering
def format_docs_with_scores(question, role):
    vstore = get_vectorstore()  # <-- This fetches the initialized vector store
    if role == "patient":
        access_filter = {"access_level": "public"}
    else:
        access_filter = {"access_level": {"$in": ["public", "staff"]}}

    try:
        docs_and_scores = vectorstore.similarity_search_with_score(
            question, k=4, filter=access_filter
        )
    except TypeError as e:
        print(f"[Filter Error]: {e}")
        docs_and_scores = []

    formatted = []
    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "SOP")
        page = doc.metadata.get("page", 0)

        similarity_percentage = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        
        header = f"[Doc: {source} | Page: {page + 1} | Score: {similarity_percentage:.2%}]"
        formatted.append(f"{header}\n{doc.page_content}")
    
    if not formatted and role == "patient":
        return "I'm sorry, but this specific information is restricted to internal clinical guidelines and is not available in public patient resources."
        
    return "\n\n".join(formatted)

# 4. Processing router function
def process_role_question(question, history, role):
    prompt = PATIENT_PROMPT if role == "patient" else STAFF_PROMPT
    context_str = format_docs_with_scores(question, role)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context_str, "question": question})

# --- GRADIO MULTI-PAGE UI ---
with gr.Blocks(title="Clinical Assistant Portal") as demo:

    # -------------------------------------------------------------
    # PAGE 1: LANDING / ROLE SELECTION
    # -------------------------------------------------------------
    with gr.Column(visible=True) as landing_page:
        gr.Markdown(
            """
            # 🏥 Clinical Assistant Portal
            ### Select your access level to begin:
            """
        )
        with gr.Row():
            patient_btn = gr.Button("🙋‍♂️ Visitor Portal", variant="primary", size="lg")
            staff_portal_btn = gr.Button("🩺 Healthcare Staff Portal", variant="secondary", size="lg")
    
    # -------------------------------------------------------------
    # PAGE 2: STAFF LOGIN SCREEN
    # -------------------------------------------------------------
    with gr.Column(visible=False) as staff_login_page:
        gr.Markdown("## 🔒 Staff Authentication Required")
        gr.Markdown("Please enter your credentials to access restricted clinical SOPs.")
        
        with gr.Row():
            username_input = gr.Textbox(label="Staff Username", placeholder="e.g., doctor_smith")
        with gr.Row():
            password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            
        with gr.Row():
            login_submit_btn = gr.Button("Login", variant="primary")
            back_btn_login = gr.Button("⬅️ Back")
            
        login_error_msg = gr.Markdown("", visible=False)

    # -------------------------------------------------------------
    # PAGE 3: PATIENT CHAT INTERFACE
    # -------------------------------------------------------------
    with gr.Column(visible=False) as patient_page:
        with gr.Row():
            gr.Markdown("## Visitor Portal (Public Guidelines)")
            back_btn_p = gr.Button("⬅️ Back")
            
        gr.ChatInterface(
            fn=lambda q, h: process_role_question(q, h, role="patient"), # <-- FIXED: Now correctly passes role="patient"
            chatbot=gr.Chatbot(
                value=[{"role": "assistant", "content": "Hello! How can I help answer your health guideline questions today?"}]
            )
        )

    # -------------------------------------------------------------
    # PAGE 4: STAFF CHAT INTERFACE
    # -------------------------------------------------------------
    with gr.Column(visible=False) as staff_page:
        with gr.Row():
            gr.Markdown("## 🩺 Healthcare Staff Portal (Full Clinical SOPs)")
            back_btn_s = gr.Button("⬅️ Back")
            
        gr.ChatInterface(
            fn=lambda q, h: process_role_question(q, h, role="staff"),
            chatbot=gr.Chatbot(
                value=[{"role": "assistant", "content": "Welcome, Staff. How may I help you with the clinical guidelines today?"}]
            )
        )

    # -------------------------------------------------------------
    # NAVIGATION & AUTHENTICATION LOGIC
    # -------------------------------------------------------------
    
    # Click Patient from Landing -> Go to Patient Page (Single Clean Binding)
    patient_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        outputs=[landing_page, staff_login_page, patient_page, staff_page]
    )
    
    # Click Staff from Landing -> Go to Staff Login Page
    staff_portal_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        outputs=[landing_page, staff_login_page, patient_page, staff_page]
    )

    # Handle Staff Login Validation
    def authenticate_staff(username, password):
        VALID_USER = "admin"
        VALID_PASS = "clinical123"
        
        if username == VALID_USER and password == VALID_PASS:
            return (
                gr.update(visible=False),  # Hide login page
                gr.update(visible=True),   # Show staff chat page
                gr.update(value="", visible=False),  # Clear error message
                gr.update(value=""),       # Clear username textbox
                gr.update(value="")        # Clear password textbox
            )
        else:
            return (
                gr.update(visible=True),   # Keep login page
                gr.update(visible=False),  # Keep chat hidden
                gr.update(value="❌ **Invalid username or password. Please try again.**", visible=True),
                gr.update(),               # Leave username as is
                gr.update()                # Leave password as is
            )

    login_submit_btn.click(
        authenticate_staff,
        inputs=[username_input, password_input],
        outputs=[staff_login_page, staff_page, login_error_msg, username_input, password_input]
    )

    # Shared clear credentials state for back buttons
    clear_creds = [
        gr.update(visible=True),   # Show landing page
        gr.update(visible=False),  # Hide login page
        gr.update(visible=False),  # Hide patient page
        gr.update(visible=False),  # Hide staff page
        gr.update(value="", visible=False), # Clear error message
        gr.update(value=""),       # Clear username textbox
        gr.update(value="")        # Clear password textbox
    ]

    back_btn_login.click(
        lambda: clear_creds,
        outputs=[landing_page, staff_login_page, patient_page, staff_page, login_error_msg, username_input, password_input]
    )
    back_btn_p.click(
        lambda: clear_creds,
        outputs=[landing_page, staff_login_page, patient_page, staff_page, login_error_msg, username_input, password_input]
    )
    back_btn_s.click(
        lambda: clear_creds,
        outputs=[landing_page, staff_login_page, patient_page, staff_page, login_error_msg, username_input, password_input]
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
