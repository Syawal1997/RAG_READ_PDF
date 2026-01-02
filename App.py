import streamlit as st
import os
import tempfile
from pathlib import Path
import sys
from typing import List, Dict, Any
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🤖 RAG PDF Chat System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #DBEAFE;
        border-left: 4px solid #3B82F6;
    }
    .bot-message {
        background-color: #F3F4F6;
        border-left: 4px solid #10B981;
    }
    .source-card {
        background-color: #FEF3C7;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #F59E0B;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 RAG PDF Chat System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload PDF files and chat with them using Gemini AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    # Model selection
    model = st.selectbox(
        "Model",
        ["gemini-pro", "gemini-pro-vision"],
        index=0
    )
    
    # Chunk settings
    st.subheader("📄 Processing Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    
    # Search settings
    st.subheader("🔍 Search Settings")
    k_results = st.slider("Number of Results", 1, 10, 5)
    
    # GitHub info
    st.markdown("---")
    st.markdown("### 📦 GitHub Repository")
    st.markdown("""
    [View on GitHub](https://github.com/yourusername/pdf-rag-system)
    
    Files will be saved to:
    `./data/processed/`
    """)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# Main content area
tab1, tab2, tab3 = st.tabs(["📤 Upload PDFs", "💬 Chat", "📊 Analytics"])

with tab1:
    st.header("Upload PDF Files")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or multiple PDF files"
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Files", use_container_width=True):
                with st.spinner("Processing PDFs..."):
                    try:
                        # Save uploaded files temporarily
                        temp_dir = tempfile.mkdtemp()
                        pdf_paths = []
                        
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(temp_path)
                        
                        # Initialize RAG system
                        from rag_system import RAGSystem
                        
                        st.session_state.rag_system = RAGSystem(api_key)
                        total_chunks = st.session_state.rag_system.process_pdfs(pdf_paths)
                        
                        st.session_state.processed_files = [
                            os.path.basename(p) for p in pdf_paths
                        ]
                        
                        st.success(f"✅ Processed {len(uploaded_files)} files with {total_chunks} chunks!")
                        
                        # Cleanup
                        import shutil
                        shutil.rmtree(temp_dir)
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        with col2:
            if st.session_state.processed_files:
                st.subheader("Processed Files")
                for file in st.session_state.processed_files:
                    st.info(f"📄 {file}")

with tab2:
    st.header("Chat with Your Documents")
    
    if not st.session_state.processed_files:
        st.warning("⚠️ Please upload and process PDF files first in the 'Upload PDFs' tab.")
    elif not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message.get("sources"):
                    with st.expander("📚 View Sources"):
                        for source in message["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{source['source']}</strong> (Page {source['page']})<br>
                                <small>{source['text_preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_system.query(
                            prompt, 
                            k=k_results
                        )
                        
                        st.markdown(response['answer'])
                        
                        if response['sources']:
                            with st.expander(f"📚 Sources ({len(response['sources'])})"):
                                for source in response['sources']:
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>{source['source']}</strong> (Page {source['page']})<br>
                                        <small>{source['text_preview']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": response['sources']
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

with tab3:
    st.header("Analytics & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Statistics")
        
        if st.session_state.processed_files:
            st.metric("Files Processed", len(st.session_state.processed_files))
            st.metric("Chat Messages", len(st.session_state.messages))
        else:
            st.info("No files processed yet")
    
    with col2:
        st.subheader("⚙️ System Info")
        st.write(f"**Model:** {model}")
        st.write(f"**Chunk Size:** {chunk_size}")
        st.write(f"**Chunk Overlap:** {chunk_overlap}")
        st.write(f"**Search Results:** {k_results}")
    
    # Export data section
    st.subheader("📤 Export Data")
    
    if st.button("Export Chat History", use_container_width=True):
        if st.session_state.messages:
            chat_history = {
                "export_date": datetime.now().isoformat(),
                "files": st.session_state.processed_files,
                "messages": st.session_state.messages
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_history, indent=2, ensure_ascii=False),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("No chat history to export")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "Built with Streamlit & Gemini API • "
    "<a href='https://github.com/yourusername/pdf-rag-system' target='_blank'>GitHub Repo</a>"
    "</div>",
    unsafe_allow_html=True
)
