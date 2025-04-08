import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client
import re
# Updated Pinecone import
from pinecone import Pinecone
import openai
import uuid
from datetime import datetime
import tempfile
# Updated LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# Remove the LangChain Pinecone import as we're using the SDK directly

# Load environment variables
load_dotenv()

# Initialize clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize Pinecone with new API
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
pinecone_index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to extract token from URL
def extract_token_from_url(url):
    match = re.search(r'access_token=([^&]+)', url)
    if match:
        return match.group(1)
    return None

# Main app
def main():
    st.set_page_config(page_title="Translation Assistant", layout="wide")
    
    # Add a form for manual token entry
    if "user" not in st.session_state:
        st.title("Translation Assistant")
        
        # Check if we need to show the token entry form
        if st.session_state.get('show_token_form', False):
            st.info("If you were redirected from email verification, please paste the entire URL here:")
            token_url = st.text_input("Verification URL", key="token_url")
            
            if st.button("Submit"):
                if token_url:
                    token = extract_token_from_url(token_url)
                    if token:
                        try:
                            user = supabase.auth.get_user(token)
                            st.session_state.user = user.user
                            st.session_state.pop('show_token_form', None)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Authentication error: {str(e)}")
                    else:
                        st.error("Could not extract token from URL")
            
            if st.button("Back to Login"):
                st.session_state.pop('show_token_form', None)
                st.rerun()
        else:
            # Show regular auth page
            show_auth_page()
    else:
        show_main_app()

# Authentication page
def show_auth_page():
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Resend Verification"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                try:
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = response.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
    
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                try:
                    response = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Account created! Please check your email to verify.")
                except Exception as e:
                    st.error(f"Sign up failed: {str(e)}")
    
    with tab3:
        with st.form("resend_verification_form"):
            email = st.text_input("Email")
            submit = st.form_submit_button("Resend Verification Email")
            
            if submit:
                try:
                    # Resend verification email
                    supabase.auth.resend_signup_email({"email": email})
                    st.success(f"Verification email resent to {email}. Please check your inbox and spam folder.")
                except Exception as e:
                    st.error(f"Failed to resend verification: {str(e)}")
        
        st.write("If you've verified your email but were redirected to an error page:")
        if st.button("Enter Verification Token"):
            st.session_state.show_token_form = True
            st.rerun()

# Main application
def show_main_app():
    st.title("Translation Assistant")
    
    # Sidebar
    with st.sidebar:
        st.button("Logout", on_click=logout)
        st.header("Upload Reference Documents")
        
        uploaded_file = st.file_uploader("Upload past translations", 
                                        type=["txt", "pdf", "docx"], 
                                        accept_multiple_files=False)
        
        source_lang = st.selectbox("Source Language", ["English", "Spanish", "French", "German", "Chinese"])
        target_lang = st.selectbox("Target Language", ["English", "Spanish", "French", "German", "Chinese"])
        
        if uploaded_file and st.button("Process Document"):
            process_document(uploaded_file, source_lang, target_lang)
    
    # Main content area
    st.header("Translation")
    
    # Get or initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for new translation
    user_input = st.chat_input("Enter text to translate...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate translation with RAG
        with st.chat_message("assistant"):
            with st.spinner("Translating..."):
                translation = translate_with_rag(user_input, source_lang, target_lang)
                st.write(translation)
                
        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": translation})
        
        # Save conversation to Supabase
        save_conversation(user_input, translation, source_lang, target_lang)

# Helper functions
def logout():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

def process_document(file, source_lang, target_lang):
    try:
        with st.spinner("Processing document..."):
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name
            
            # Load document based on file type
            if file.name.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Process and upload chunks directly to Pinecone
            for i, chunk in enumerate(chunks):
                # Create a unique ID for this vector
                vector_id = f"{st.session_state.user.id}-{file.name}-{i}-{uuid.uuid4()}"
                
                # Create metadata
                metadata = {
                    "text": chunk.page_content,
                    "source": file.name,
                    "user_id": st.session_state.user.id,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Get embedding for the chunk
                vector = embeddings.embed_documents([chunk.page_content])[0]
                
                # Upload to Pinecone directly
                pinecone_index.upsert(
                    vectors=[
                        {
                            "id": vector_id,
                            "values": vector,
                            "metadata": metadata
                        }
                    ]
                )
            
            # Store document metadata in Supabase
            supabase.table("documents").insert({
                "user_id": st.session_state.user.id,
                "filename": file.name,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "upload_date": datetime.now().isoformat()
            }).execute()
            
            # Clean up temp file
            os.unlink(file_path)
            
            st.sidebar.success(f"Document '{file.name}' processed successfully!")
    except Exception as e:
        st.sidebar.error(f"Error processing document: {str(e)}")
        st.sidebar.error(f"Error details: {type(e).__name__}: {str(e)}")

def translate_with_rag(text, source_lang, target_lang):
    try:
        # Create embedding for the query
        query_embedding = embeddings.embed_query(text)
        
        # Search directly with Pinecone
        search_results = pinecone_index.query(
            vector=query_embedding,
            filter={"user_id": st.session_state.user.id},
            top_k=3,
            include_metadata=True
        )
        
        # Extract context from search results
        context = ""
        if search_results.matches:
            context = "\n\n".join([match.metadata.get("text", "") for match in search_results.matches if "text" in match.metadata])
        
        # Call OpenRouter API for translation
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {"role": "system", "content": f"""You are a professional translator from {source_lang} to {target_lang}. 
                Use the following past translations as reference to maintain consistency in terminology and style:
                
                {context}
                
                Translate the user's text maintaining the same tone, terminology, and formatting as the reference translations."""},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Translation error: {str(e)}"

def save_conversation(source_text, translated_text, source_lang, target_lang):
    try:
        # Generate a thread ID if not exists
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        
        # Ensure user is authenticated
        if "user" not in st.session_state or not st.session_state.user:
            st.error("User not authenticated. Please log in again.")
            return
            
        # Get the user ID from session state
        user_id = st.session_state.user.id
        
        # Debug output
        st.write(f"Saving with user ID: {user_id}")
        
        # Save to Supabase
        result = supabase.table("translations").insert({
            "user_id": user_id,
            "thread_id": st.session_state.thread_id,
            "source_text": source_text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "timestamp": datetime.now().isoformat()
        }).execute()
        
        # Check if there was an error in the response
        if hasattr(result, 'error') and result.error:
            st.error(f"Error saving translation: {result.error}")
            
    except Exception as e:
        st.error(f"Error saving translation: {str(e)}")
        # Add debug info
        st.write(f"Exception details: {type(e).__name__}")

if __name__ == "__main__":
    main()
