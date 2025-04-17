import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client
import re
# Import pinecone in a way that works with both v1 and v2
import importlib.util
import sys

# Define a function to initialize Pinecone based on the available version
def initialize_pinecone():
    # Check if pinecone is installed
    if importlib.util.find_spec("pinecone") is None:
        raise ImportError("Pinecone package is not installed")

    # Import the package
    import pinecone

    # Check if it's the new version (v2) with Pinecone class
    if hasattr(pinecone, "Pinecone"):
        # It's v2
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX"))
        return index
    else:
        # It's v1
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index = pinecone.Index(os.getenv("PINECONE_INDEX"))
        return index

# Initialize Pinecone
try:
    pinecone_index = initialize_pinecone()
except Exception as e:
    print(f"Error initializing Pinecone: {str(e)}")
    # Create a dummy index for development without Pinecone
    class DummyIndex:
        def query(self, *args, **kwargs):
            class DummyResponse:
                matches = []
            return DummyResponse()
        def upsert(self, *args, **kwargs):
            pass
    pinecone_index = DummyIndex()
import openai
import uuid
from datetime import datetime
import tempfile
# Use simpler document loaders without LangChain
import PyPDF2
import docx2txt
# Use OpenAI directly for embeddings
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Pinecone is already initialized above

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to extract token from URL
def extract_token_from_url(url):
    # This regex will match access_token in both fragment (#) and query parameter (?) formats
    match = re.search(r'[#?&]access_token=([^&]+)', url)
    if match:
        return match.group(1)

    # Log the URL format for debugging (without exposing the actual token)
    print(f"URL format: {url[:url.find('=')+1]}...")
    return None

# Function to ensure proper encoding of Japanese text
def ensure_proper_encoding(text):
    """Ensure text is properly encoded, especially for Japanese characters."""
    try:
        # If text is already a string, just return it
        if isinstance(text, str):
            return text
        # If it's bytes, decode it properly
        if isinstance(text, bytes):
            return text.decode('utf-8')
        # Otherwise, convert to string
        return str(text)
    except Exception as e:
        # Log the error
        print(f"Encoding error: {str(e)}")
        # If any error occurs, return the original text
        return text

# Main app
def main():
    st.set_page_config(page_title="Translation Assistant", layout="wide")

    # Set page configuration for proper UTF-8 encoding
    st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Noto Sans JP', sans-serif;
        }
    </style>
    ''', unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # Try to restore user session from cookies or local storage
        try:
            # Check for auth redirect in URL query parameters
            if "access_token" in st.query_params:
                token = st.query_params["access_token"]
                try:
                    user = supabase.auth.get_user(token)
                    st.session_state.user = user.user
                    # Clear query params to avoid reprocessing
                    st.query_params.clear()
                except Exception as e:
                    st.error(f"Authentication error from redirect: {str(e)}")
            else:
                # Check if there's a valid session in Supabase
                session = supabase.auth.get_session()
                if session and hasattr(session, 'user'):
                    st.session_state.user = session.user

                # Also try to load user documents from local storage
                try:
                    # Create a local file to store user data if it doesn't exist
                    user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
                    os.makedirs(user_data_dir, exist_ok=True)

                    user_id = session.user.id
                    user_data_file = os.path.join(user_data_dir, f"{user_id}_documents.json")

                    if os.path.exists(user_data_file):
                        with open(user_data_file, 'r') as f:
                            import json
                            user_documents = json.load(f)
                            st.session_state.user_documents = user_documents
                except Exception as e:
                    # Silently handle errors in loading local data
                    pass
        except Exception:
            # If there's an error, we'll just continue with login flow
            pass

    # Add auto-login for development (remove in production)
    if "user" not in st.session_state and os.getenv("STREAMLIT_ENV") == "development":
        # Create a dummy user for development with a FIXED ID to ensure persistence
        dev_user_id = "dev-user-fixed-id-123"
        st.session_state.user = {"id": dev_user_id, "email": "dev@example.com"}

        # Try to load documents from local storage
        try:
            user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
            user_data_file = os.path.join(user_data_dir, f"{dev_user_id}_documents.json")

            if os.path.exists(user_data_file):
                with open(user_data_file, 'r') as f:
                    import json
                    try:
                        st.session_state.user_documents = json.load(f)
                    except:
                        st.session_state.user_documents = []
            else:
                st.session_state.user_documents = []
        except Exception:
            st.session_state.user_documents = []

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

        # Create tabs for upload and manage documents
        upload_tab, manage_tab = st.tabs(["Upload Documents", "Manage Documents"])

        with upload_tab:
            st.header("Upload Reference Documents")

            uploaded_files = st.file_uploader("Upload past translations",
                                            type=["txt", "pdf", "docx"],
                                            accept_multiple_files=True)

            source_lang = st.selectbox("Source Language", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])
            target_lang = st.selectbox("Target Language", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])

            if uploaded_files and st.button("Process Documents"):
                for uploaded_file in uploaded_files:
                    process_document(uploaded_file, source_lang, target_lang)

        with manage_tab:
            st.header("Manage Reference Documents")
            if st.button("Refresh Document List"):
                st.rerun()

            # Fetch and display user's documents
            documents = fetch_user_documents()

            if not documents:
                st.info("No reference documents found. Upload some documents to get started!")
            else:
                for doc in documents:
                    with st.expander(f"{doc['filename']} ({doc['upload_date'][:10]})"):
                        st.write(f"Source Language: {doc['source_lang']}")
                        st.write(f"Target Language: {doc['target_lang']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Delete", key=f"delete_{doc['id']}"):
                                delete_document(doc['id'])
                                st.success(f"Document '{doc['filename']}' deleted successfully!")
                                st.rerun()

                        with col2:
                            if st.button("Update Languages", key=f"update_{doc['id']}"):
                                st.session_state.update_doc_id = doc['id']
                                st.session_state.update_doc_name = doc['filename']
                                st.rerun()

            # Show update form if a document is selected for update
            if "update_doc_id" in st.session_state:
                st.subheader(f"Update {st.session_state.update_doc_name}")
                new_source_lang = st.selectbox("New Source Language", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])
                new_target_lang = st.selectbox("New Target Language", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])

                if st.button("Save Changes"):
                    update_document_languages(st.session_state.update_doc_id, new_source_lang, new_target_lang)
                    st.success(f"Document '{st.session_state.update_doc_name}' updated successfully!")
                    del st.session_state.update_doc_id
                    del st.session_state.update_doc_name
                    st.rerun()

                if st.button("Cancel"):
                    del st.session_state.update_doc_id
                    del st.session_state.update_doc_name
                    st.rerun()

    # Main content area
    st.header("Translation")

    # Get or initialize chat threads
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = {}

        # Try to load chat threads from local storage
        if "user" in st.session_state and st.session_state.user:
            try:
                user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id
                user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
                chat_threads_file = os.path.join(user_data_dir, f"{user_id}_chat_threads.json")

                if os.path.exists(chat_threads_file):
                    with open(chat_threads_file, 'r') as f:
                        import json
                        try:
                            st.session_state.chat_threads = json.load(f)
                        except:
                            st.session_state.chat_threads = {}
            except Exception:
                # If loading fails, start with empty chat threads
                st.session_state.chat_threads = {}

    # Add a "New Chat" button in the sidebar
    if st.sidebar.button("New Chat"):
        new_thread_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_thread_id
        st.session_state.chat_threads[new_thread_id] = {
            "name": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "messages": []
        }
        st.rerun()

    # Display chat thread selection in sidebar
    st.sidebar.subheader("Your Conversations")

    # Create columns for each thread to display thread name and rename button
    for thread_id, thread_data in st.session_state.chat_threads.items():
        col1, col2 = st.sidebar.columns([4, 1])

        # Thread selection button
        if col1.button(thread_data["name"], key=f"thread_{thread_id}"):
            st.session_state.current_thread_id = thread_id
            st.rerun()

        # Rename button
        if col2.button("✏️", key=f"rename_{thread_id}"):
            st.session_state.rename_thread_id = thread_id
            st.session_state.rename_thread_name = thread_data["name"]
            st.rerun()

    # Show rename form if a thread is selected for renaming
    if "rename_thread_id" in st.session_state:
        st.sidebar.subheader("Rename Conversation")
        new_name = st.sidebar.text_input("New name", value=st.session_state.rename_thread_name)

        col1, col2 = st.sidebar.columns(2)
        if col1.button("Save"):
            # Update the thread name
            st.session_state.chat_threads[st.session_state.rename_thread_id]["name"] = new_name
            # Remove the rename state
            st.session_state.pop("rename_thread_id", None)
            st.session_state.pop("rename_thread_name", None)
            st.rerun()

        if col2.button("Cancel"):
            # Remove the rename state without saving
            st.session_state.pop("rename_thread_id", None)
            st.session_state.pop("rename_thread_name", None)
            st.rerun()



    # Initialize current thread ID if it doesn't exist
    if "current_thread_id" not in st.session_state or not st.session_state.chat_threads:
        new_thread_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_thread_id
        st.session_state.chat_threads[new_thread_id] = {
            "name": f"New conversation",  # Default name until content is available
            "messages": []
        }

    # Ensure the current thread exists in chat_threads
    if st.session_state.current_thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads[st.session_state.current_thread_id] = {
            "name": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "messages": []
        }

    # Get current chat messages
    current_messages = st.session_state.chat_threads[st.session_state.current_thread_id]["messages"]

    # Display current chat thread name
    st.subheader(st.session_state.chat_threads[st.session_state.current_thread_id]["name"])

    # Display chat history
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input for new translation
    user_input = st.chat_input("Enter text to translate...")

    if user_input:
        # Add user message to current chat thread
        st.session_state.chat_threads[st.session_state.current_thread_id]["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate translation with RAG
        with st.chat_message("assistant"):
            with st.spinner("Translating..."):
                # Normal translation for all texts
                translation = translate_with_rag(user_input, source_lang, target_lang)

                # Display with proper encoding - use HTML entities for Japanese characters
                # Convert to HTML entities to ensure proper display
                import html
                safe_translation = html.escape(translation)

                # Display as read-only text (not editable)
                st.markdown(f"""<div style="font-family: 'Noto Sans JP', sans-serif; font-size: 16px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                {safe_translation}
                </div>""", unsafe_allow_html=True)

        # Add assistant message to current chat thread
        st.session_state.chat_threads[st.session_state.current_thread_id]["messages"].append({"role": "assistant", "content": translation})

        # Generate a title for the thread if it's the first message
        if len(st.session_state.chat_threads[st.session_state.current_thread_id]["messages"]) == 2 and st.session_state.chat_threads[st.session_state.current_thread_id]["name"] == "New conversation":
            # Generate a title based on the first message
            thread_title = generate_thread_title(user_input)
            st.session_state.chat_threads[st.session_state.current_thread_id]["name"] = thread_title

        # Save conversation to local storage
        save_conversation(user_input, translation, source_lang, target_lang, st.session_state.current_thread_id)

# Helper functions
def logout():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

# Function to generate a title for a chat thread based on the first message
def generate_thread_title(first_message):
    try:
        # Create OpenRouter client
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # Generate a title using the same model as for translations
        response = client.chat.completions.create(
            model="arliai/qwq-32b-arliai-rpr-v1:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates short, descriptive titles (3-5 words) for conversations based on the first message. The title should be concise and reflect the main topic or intent of the message."},
                {"role": "user", "content": f"Generate a short title (3-5 words) for a conversation that starts with this message: '{first_message}'. Return ONLY the title, nothing else."}
            ],
            temperature=0.3,
            max_tokens=20
        )

        # Extract the title from the response
        if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and response.choices[0].message:
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    # Clean up the title (remove quotes, etc.)
                    title = response.choices[0].message.content.strip().strip('"').strip("'").strip()
                    return title

        # Fallback title if generation fails
        return f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        # If there's any error, use a timestamp-based title
        print(f"Error generating title: {str(e)}")
        return f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

def process_document(file, source_lang, target_lang):
    try:
        with st.spinner(f"Processing document {file.name}..."):
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name

            # Extract text based on file type
            text = ""
            if file.name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file.name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            elif file.name.endswith('.docx'):
                text = docx2txt.process(file_path)

            # Simple text splitting (1000 chars per chunk with 100 char overlap)
            chunks = []
            chunk_size = 3000
            overlap = 300

            if len(text) <= chunk_size:
                chunks = [text]
            else:
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    chunks.append(chunk)

            # Generate a document ID
            doc_id = str(uuid.uuid4())

            # Process and upload chunks directly to Pinecone
            for i, chunk in enumerate(chunks):
                # Create a unique ID for this vector
                # Get user ID safely
                user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id
                vector_id = f"{user_id}-{doc_id}-{i}-{uuid.uuid4()}"

                # Create metadata
                metadata = {
                    "text": chunk,
                    "source": file.name,
                    "document_id": doc_id,  # Add document ID to metadata
                    "user_id": st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "timestamp": datetime.now().isoformat()
                }

                # Get embedding for the chunk using OpenAI directly
                response = openai_client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )
                vector = response.data[0].embedding

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

            # Store document metadata in Supabase with RLS handling
            try:
                # Try to use the auth client to respect RLS policies
                supabase.table("documents").insert({
                    "id": doc_id,  # Use the generated document ID
                    "user_id": st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id,
                    "filename": file.name,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "upload_date": datetime.now().isoformat()
                }).execute()
            except Exception:
                # Silently handle the database error
                pass

            # Always store in session state as a backup
            if "user_documents" not in st.session_state:
                st.session_state.user_documents = []

            # Create document metadata
            doc_metadata = {
                "id": doc_id,
                "user_id": st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id,
                "filename": file.name,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "upload_date": datetime.now().isoformat()
            }

            # Add to session state
            st.session_state.user_documents.append(doc_metadata)

            # Save to local storage for persistence
            try:
                # Create a local file to store user data if it doesn't exist
                user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
                os.makedirs(user_data_dir, exist_ok=True)

                user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id
                user_data_file = os.path.join(user_data_dir, f"{user_id}_documents.json")

                # Load existing data if file exists
                existing_docs = []
                if os.path.exists(user_data_file):
                    with open(user_data_file, 'r') as f:
                        import json
                        try:
                            existing_docs = json.load(f)
                        except:
                            existing_docs = []

                # Add new document and save
                existing_docs.append(doc_metadata)
                with open(user_data_file, 'w') as f:
                    import json
                    json.dump(existing_docs, f)
            except Exception as e:
                # Silently handle errors in saving to local storage
                pass

            # Clean up temp file
            os.unlink(file_path)

            st.success(f"Document '{file.name}' processed successfully!")
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.error(f"Error details: {type(e).__name__}: {str(e)}")

def translate_with_rag(text, source_lang, target_lang):
    try:
        # If the text contains "translate into japanese" or similar, extract the actual text to translate
        if "translate into japanese" in text.lower() or "translate to japanese" in text.lower():
            # Extract the text before the instruction
            text = text.lower().split("translate")[0].strip()
            target_lang = "Japanese"
        elif "translate into english" in text.lower() or "translate to english" in text.lower():
            # Extract the text before the instruction
            text = text.lower().split("translate")[0].strip()
            target_lang = "English"

        # Create embedding for the query using OpenAI directly
        query_response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        # Search directly with Pinecone
        search_results = pinecone_index.query(
            vector=query_embedding,
            filter={"user_id": st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id},
            top_k=3,
            include_metadata=True
        )

        # Extract context from search results
        context = ""
        if hasattr(search_results, 'matches') and search_results.matches:
            context = "\n\n".join([match.metadata.get("text", "") for match in search_results.matches if "text" in match.metadata])

        # Use OpenAI directly for translation instead of OpenRouter
        # This is more reliable and has better Japanese support

        # Prepare a more detailed system prompt with stronger emphasis on matching reference style
        system_prompt = f"""You are a professional translator from {source_lang} to {target_lang} who EXACTLY matches the style, tone, and terminology of reference translations.

        CRITICAL INSTRUCTION: Your PRIMARY goal is to PRECISELY mirror the style, word choice, and tone of the reference translations. Accuracy is important but SECONDARY to matching the reference style.

        Reference translations to match style and terminology from:
        {context}

        STRICT Translation guidelines:
        1. EXACTLY copy the terminology used in the reference translations - use the SAME words for the same concepts
        2. EXACTLY match the tone (formal/informal) used in the reference translations
        3. EXACTLY match the sentence structure patterns when possible
        4. EXACTLY preserve all formatting, including paragraph breaks and punctuation
        5. If specific words appear in the reference, use those EXACT same words in your translation
        6. For example, if the reference text translates "shop" as "ブティック", you MUST use "ブティック" for "shop"
        7. Follow the terminology from the reference word by word with extreme precision

        For Japanese specifically:
        - Use the EXACT SAME level of keigo/politeness as in the reference
        - Use the EXACT SAME kanji/hiragana/katakana choices as in the reference
        - Use 「ブティック」 instead of 「店」 for 'shop' if that appears in the reference
        - Use 「アトリエ」 instead of 「店」 for 'store' if that appears in the reference
        - Use 「お客様」 instead of 「客」 for 'customer' if that appears in the reference
        - If the reference uses 「角」 for 'corner', use that exact term
        - If the reference uses specific particles (は, が, を, etc.), use the SAME particles
        - If the reference uses specific verb forms or tenses, match them exactly

        IMPORTANT: You MUST translate the text to {target_lang}. Do not repeat the original text.

        Your translation should be INDISTINGUISHABLE in style from the reference translations.
        """

        # Call OpenRouter API for translation
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # For Japanese translations, we need to be extra careful with encoding
        if source_lang == "Japanese" or target_lang == "Japanese":
            # Add explicit encoding instructions
            user_message = f"Translate this text from {source_lang} to {target_lang}. Ensure proper UTF-8 encoding for all Japanese characters: {text}"

            # Use a lower temperature for more consistent results with Japanese
            response = client.chat.completions.create(
                model="arliai/qwq-32b-arliai-rpr-v1:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1  # Even lower temperature for Japanese
            )
        else:
            # For non-Japanese translations
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3-base:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this text from {source_lang} to {target_lang}: {text}"}
                ],
                temperature=0.2  # Lower temperature for more consistent results
            )

        # Ensure proper encoding of the translation result
        if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and response.choices[0].message:
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    result = ensure_proper_encoding(response.choices[0].message.content)
                    return result
                else:
                    return "Translation error: No content in response message"
            else:
                return "Translation error: No message in response"
        else:
            return "Translation error: Invalid response format"
    except Exception as e:
        return f"Translation error: {str(e)}"

def save_conversation(source_text, translated_text, source_lang, target_lang, thread_id=None):
    try:
        # Use provided thread_id or current thread_id
        if thread_id is None and "current_thread_id" in st.session_state:
            thread_id = st.session_state.current_thread_id
        elif thread_id is None:
            thread_id = str(uuid.uuid4())

        # Ensure user is authenticated
        if "user" not in st.session_state or not st.session_state.user:
            st.error("User not authenticated. Please log in again.")
            return

        # Get the user ID from session state
        user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id

        # Save to local storage
        try:
            # Create a local file to store chat threads if it doesn't exist
            user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
            os.makedirs(user_data_dir, exist_ok=True)

            # Save chat threads to file
            chat_threads_file = os.path.join(user_data_dir, f"{user_id}_chat_threads.json")

            # Convert chat threads to serializable format
            serializable_threads = {}
            for t_id, thread_data in st.session_state.chat_threads.items():
                serializable_threads[t_id] = thread_data

            # Save to file
            with open(chat_threads_file, 'w') as f:
                import json
                json.dump(serializable_threads, f)

            # Also save individual translation for backward compatibility
            translation_file = os.path.join(user_data_dir, f"{user_id}_translations.json")

            # Load existing translations if file exists
            existing_translations = []
            if os.path.exists(translation_file):
                with open(translation_file, 'r') as f:
                    import json
                    try:
                        existing_translations = json.load(f)
                    except:
                        existing_translations = []

            # Add new translation
            translation_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "thread_id": thread_id,
                "source_text": source_text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.now().isoformat()
            }

            existing_translations.append(translation_data)

            # Save to file
            with open(translation_file, 'w') as f:
                import json
                json.dump(existing_translations, f)

        except Exception as storage_error:
            st.error(f"Error saving translation to local storage: {str(storage_error)}")

    except Exception as e:
        st.error(f"Error saving translation: {str(e)}")
        # Add debug info
        st.write(f"Exception details: {type(e).__name__}")

# Function to fetch user's documents from Supabase or session state
def fetch_user_documents():
    try:
        # Ensure user is authenticated
        if "user" not in st.session_state or not st.session_state.user:
            st.error("User not authenticated. Please log in again.")
            return []

        # Get the user ID from session state
        user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id

        # Combine documents from both sources
        all_docs = []

        # First try to fetch from Supabase
        try:
            result = supabase.table("documents").select("*").eq("user_id", user_id).order("upload_date", desc=True).execute()
            if hasattr(result, 'data') and result.data:
                all_docs.extend(result.data)
        except Exception:
            # Silently handle database errors
            pass

        # Also get documents from session state
        if "user_documents" in st.session_state:
            all_docs.extend(st.session_state.user_documents)

        # Sort all documents by upload date (newest first)
        if all_docs:
            sorted_docs = sorted(all_docs,
                               key=lambda x: x.get('upload_date', ''),
                               reverse=True)

            # Remove duplicates (prefer session state versions)
            unique_docs = []
            seen_ids = set()

            for doc in sorted_docs:
                doc_id = doc.get('id')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

            return unique_docs

        # If no documents found anywhere
        return []

    except Exception as e:
        # Only show error for non-database related issues
        st.error(f"Error fetching documents: {str(e)}")
        return []

# Function to delete a document and its vectors from Pinecone
def delete_document(doc_id):
    try:
        # Ensure user is authenticated
        if "user" not in st.session_state or not st.session_state.user:
            st.error("User not authenticated. Please log in again.")
            return False

        # Get the user ID from session state
        user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id

        # Try to delete from Supabase first
        try:
            supabase.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
        except Exception:
            # Silently handle database errors
            pass

        # Also delete from session state if it exists there
        if "user_documents" in st.session_state:
            st.session_state.user_documents = [
                doc for doc in st.session_state.user_documents
                if doc.get('id') != doc_id
            ]

        # Delete from local storage for persistence
        try:
            # Get the local storage path
            user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
            user_data_file = os.path.join(user_data_dir, f"{user_id}_documents.json")

            if os.path.exists(user_data_file):
                # Load existing data
                with open(user_data_file, 'r') as f:
                    import json
                    try:
                        existing_docs = json.load(f)
                        # Filter out the document to delete
                        existing_docs = [doc for doc in existing_docs if doc.get('id') != doc_id]
                        # Save the updated list
                        with open(user_data_file, 'w') as f:
                            json.dump(existing_docs, f)
                    except:
                        pass
        except Exception:
            # Silently handle errors in local storage operations
            pass

        # Delete vectors from Pinecone
        # First, fetch vectors with this document_id
        response = pinecone_index.query(
            vector=[0] * 1536,  # Dummy vector for metadata filtering
            filter={"document_id": doc_id, "user_id": user_id},
            top_k=1000,  # Get as many as possible
            include_metadata=True
        )

        # Extract vector IDs
        if response.matches:
            vector_ids = [match.id for match in response.matches]

            # Delete vectors in batches of 100
            for i in range(0, len(vector_ids), 100):
                batch = vector_ids[i:i+100]
                pinecone_index.delete(ids=batch)

        return True

    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

# Function to update document languages
def update_document_languages(doc_id, new_source_lang, new_target_lang):
    try:
        # Ensure user is authenticated
        if "user" not in st.session_state or not st.session_state.user:
            st.error("User not authenticated. Please log in again.")
            return False

        # Get the user ID from session state
        user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id

        # Try to update in Supabase first
        try:
            supabase.table("documents").update({
                "source_lang": new_source_lang,
                "target_lang": new_target_lang
            }).eq("id", doc_id).eq("user_id", user_id).execute()
        except Exception:
            # Silently handle database errors
            pass

        # Also update in session state if it exists there
        if "user_documents" in st.session_state:
            for doc in st.session_state.user_documents:
                if doc.get('id') == doc_id:
                    doc['source_lang'] = new_source_lang
                    doc['target_lang'] = new_target_lang

        # Update in local storage for persistence
        try:
            # Get the local storage path
            user_data_dir = os.path.join(os.path.expanduser("~"), ".translation_app")
            user_data_file = os.path.join(user_data_dir, f"{user_id}_documents.json")

            if os.path.exists(user_data_file):
                # Load existing data
                with open(user_data_file, 'r') as f:
                    import json
                    try:
                        existing_docs = json.load(f)
                        # Update the document
                        for doc in existing_docs:
                            if doc.get('id') == doc_id:
                                doc['source_lang'] = new_source_lang
                                doc['target_lang'] = new_target_lang
                        # Save the updated list
                        with open(user_data_file, 'w') as f:
                            json.dump(existing_docs, f)
                    except:
                        pass
        except Exception:
            # Silently handle errors in local storage operations
            pass

        # Update vectors in Pinecone
        # First, fetch vectors with this document_id
        response = pinecone_index.query(
            vector=[0] * 1536,  # Dummy vector for metadata filtering
            filter={"document_id": doc_id, "user_id": user_id},
            top_k=1000,  # Get as many as possible
            include_metadata=True
        )

        # Update each vector's metadata
        if response.matches:
            for match in response.matches:
                # Get the existing metadata
                metadata = match.metadata

                # Update the language fields
                metadata["source_lang"] = new_source_lang
                metadata["target_lang"] = new_target_lang

                # Update the vector in Pinecone
                pinecone_index.update(
                    id=match.id,
                    values=None,  # Keep the existing vector
                    metadata=metadata
                )

        return True

    except Exception as e:
        st.error(f"Error updating document: {str(e)}")
        return False

if __name__ == "__main__":
    main()
