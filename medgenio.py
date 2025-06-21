import os
import time
import glob
import pathlib
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import streamlit as st
from together import Together
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import torch  # Adicionado para verificação de GPU

# Carrega variáveis do .env
load_dotenv()

# ========== CONFIGURAÇÕES ATUALIZADAS (ClinicalBERT) ==========
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
together_API_KEY = os.getenv("together_API_KEY")

# Configurações de embeddings médicos
EMBEDDING_MODEL = "medicalai/ClinicalBERT"  # Modelo específico para saúde
MAX_WORKERS = 4
CHUNK_SIZE = 384  # Tamanho reduzido para compatibilidade com BERT
CHUNK_OVERLAP = 64

# === DIRETÓRIOS ===
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
KNOWLEDGE_FOLDER = SCRIPT_DIR / "base_conhecimento"
DB_PATH = SCRIPT_DIR / "chroma_db_clinicalbert"  # Novo diretório para evitar conflitos

# Cria as pastas se não existirem
KNOWLEDGE_FOLDER.mkdir(exist_ok=True, parents=True)
DB_PATH.mkdir(exist_ok=True, parents=True)

# ========== CLASSE TOGETHER MED ==========
class TogetherMED:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)
        self.conversation_history = []
        self.history_lock = Lock()
    
    def gerar_resposta_streaming(self, prompt, contexto="", max_tokens=MAX_TOKENS):
        try:
            context_hash = hashlib.md5(contexto.encode()).hexdigest()[:8]
            cache_key = f"{prompt[:50]}-{context_hash}"
            
            if st.session_state.get(cache_key):
                yield st.session_state[cache_key]
                return
            
            messages = [{
                "role": "system",
                "content": (
                    "Você é um assistente médico IA especializado. Diretrizes:\n"
                    "1. Respostas objetivas baseadas em evidências clínicas\n"
                    "2. Contexto fornecido:\n" + (contexto[:2000] + "..." if len(contexto) > 2000 else contexto) + "\n"
                    "3. Encaminhar para avaliação presencial quando necessário\n"
                    "4. Considerar terminologia médica especializada"
                )
            }]
            
            with self.history_lock:
                recent_history = self.conversation_history[-6:]
                messages.extend(recent_history)
                messages.append({"role": "user", "content": prompt})
            
            stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token
            
            st.session_state[cache_key] = full_response
            
            with self.history_lock:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            yield f"⚠️ Erro na geração da resposta: {str(e)}"
    
    def limpar_historico(self):
        with self.history_lock:
            self.conversation_history = []

# ========== CLASSE OCR LOADER ==========
class OCRPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        try:
            images = convert_from_path(self.file_path, first_page=0, last_page=20)
            return [
                Document(
                    page_content=pytesseract.image_to_string(image),
                    metadata={"source": self.file_path, "page": i+1}
                ) for i, image in enumerate(images)
            ]
        except Exception as e:
            st.error(f"Erro no OCR: {str(e)}")
            return []

# ========== SISTEMA DE EMBEDDINGS (ClinicalBERT) ==========
@st.cache_resource
def get_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"ClinicalBERT usando: {device.upper()}")
    
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=device
    )

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(DB_PATH))

def get_or_create_collection(name="medical_knowledge_clinicalbert"):  # Nome alterado
    client = get_chroma_client()
    embedding_function = get_embedding_function()
    
    try:
        collection = client.get_collection(name)
        st.sidebar.success("Coleção ClinicalBERT carregada")
        return collection
    except:
        st.sidebar.info("Criando nova coleção com ClinicalBERT")
        return client.create_collection(
            name=name,
            embedding_function=embedding_function
        )

# ========== PROCESSAMENTO DE DOCUMENTOS ==========
def carregar_documento(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            try:
                return PyPDFLoader(file_path).load()
            except:
                return OCRPDFLoader(file_path).load()
        elif ext == '.docx':
            return Docx2txtLoader(file_path).load()
    except Exception as e:
        st.error(f"Erro ao processar {file_path}: {str(e)}")
    return []

def processar_documento(file_path):
    try:
        docs = carregar_documento(file_path)
        if not docs:
            return None, f"Falha ao carregar: {os.path.basename(file_path)}"
        
        valid_docs = [doc for doc in docs if doc.page_content.strip()]
        if not valid_docs:
            return None, f"Documento vazio: {os.path.basename(file_path)}"
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_documents(valid_docs), None
    except Exception as e:
        return None, f"Erro crítico: {os.path.basename(file_path)} - {str(e)}"

def carregar_base():
    collection = get_or_create_collection()
    
    if collection.count() > 0:
        st.sidebar.info(f"{collection.count()} documentos na base")
        return collection
    
    arquivos = []
    for ext in ('*.pdf', '*.docx'):
        arquivos.extend(glob.glob(f"{KNOWLEDGE_FOLDER}/**/{ext}", recursive=True))
    
    if not arquivos:
        st.warning("Nenhum documento encontrado na base de conhecimento!")
        return collection
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(processar_documento, file) for file in arquivos]
        
        total = len(arquivos)
        progress_bar = st.progress(0)
        chunks = []
        erros = []
        
        for i, future in enumerate(futures):
            chunks_batch, error = future.result()
            if chunks_batch:
                chunks.extend(chunks_batch)
            if error:
                erros.append(error)
            
            progress_bar.progress((i + 1) / total)
    
    if chunks:
        batch_size = 256  # Reduzido para ClinicalBERT
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            ids = [f"doc_{i + j}" for j in range(len(batch_chunks))]
            documents = [chunk.page_content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    if erros:
        st.error(f"{len(erros)} erros durante o processamento")
        with st.expander("Detalhes dos erros"):
            for erro in erros:
                st.write(erro)
    
    return collection

# ========== ATUALIZAÇÃO INCREMENTAL ==========
def atualizar_base(file_path):
    try:
        chunks, error = processar_documento(file_path)
        if not chunks:
            return error
        
        collection = get_or_create_collection()
        existing_ids = collection.get()["ids"]
        new_id_base = f"doc_{len(existing_ids)}"
        
        batch_size = 256  # Reduzido para ClinicalBERT
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            ids = [f"{new_id_base}_{i + j}" for j in range(len(batch_chunks))]
            documents = [chunk.page_content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return f"Documento atualizado: {os.path.basename(file_path)}"
    except Exception as e:
        return f"Erro na atualização: {str(e)}"

# ========== MONITORAMENTO DE ARQUIVOS ==========
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_updated = time.time()
        self.lock = Lock()
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.pdf', '.docx')):
            current_time = time.time()
            if current_time - self.last_updated > 30:
                with self.lock:
                    self.last_updated = current_time
                    result = atualizar_base(event.src_path)
                    if result:
                        st.toast(f"ClinicalBERT: {result}")

# ========== FUNÇÕES DE BUSCA ==========
def buscar_contexto(query, collection, k=3):
    if collection.count() == 0:
        return ""
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    return "\n\n".join(results["documents"][0])

# ========== INTERFACE PRINCIPAL ==========
def main():
    st.set_page_config(
        page_title="⚕️ Médico Virtual - ClinicalBERT",
        page_icon="⚕️",
        layout="centered"
    )
    
    if 'file_watcher' not in st.session_state:
        event_handler = FileChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, path=str(KNOWLEDGE_FOLDER), recursive=True)
        observer.start()
        st.session_state.file_watcher = observer
    
    if "assistant" not in st.session_state:
        st.session_state.assistant = TogetherMED(together_API_KEY)
    
    if "knowledge_base" not in st.session_state:
        with st.spinner("🚀 Inicializando ClinicalBERT e base de conhecimento..."):
            st.session_state.knowledge_base = carregar_base()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Olá! Sou seu assistente médico com ClinicalBERT. Base de conhecimento em: `{KNOWLEDGE_FOLDER}`"
        }]
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("⚙️ Configurações Clínicas")
        
        st.metric("Documentos na base", st.session_state.knowledge_base.count())
        st.metric("Chunks de conhecimento", len(st.session_state.knowledge_base.get()["ids"]))
        
        if st.button("🔄 Recarregar Base", use_container_width=True, key="reload_btn"):
            with st.spinner("Otimizando embeddings clínicos..."):
                st.cache_resource.clear()
                st.session_state.knowledge_base = carregar_base()
                st.rerun()
        
        st.divider()
        
        n_resultados = st.slider("Trechos relevantes", 1, 5, 3, key="context_slider")
        max_tokens = st.slider("Comprimento da resposta", 256, 4096, MAX_TOKENS, key="tokens_slider")
        
        modelo_selecionado = st.selectbox(
            "Modelo de IA",
            ["DeepSeek-V3"],
            index=0,
            key="model_selector"
        )
        
        if st.button("🧹 Limpar Histórico", use_container_width=True, key="clear_hist"):
            st.session_state.assistant.limpar_historico()
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Histórico limpo! Como posso ajudar?"
            }]
            st.rerun()
        
        st.divider()
        st.caption(f"Embeddings: ClinicalBERT")
        st.caption(f"Chunk Size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
        st.caption("Desenvolvido com Together API")

    # ========== ÁREA PRINCIPAL ==========
    st.title("⚕️ Médico Virtual - ClinicalBERT")
    st.caption("Sistema de apoio clínico com embeddings médicos especializados")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Digite sua dúvida médica"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        n_resultados = st.session_state.get("context_slider", 3)
        contexto = buscar_contexto(prompt, st.session_state.knowledge_base, k=n_resultados)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            max_tokens = st.session_state.get("tokens_slider", MAX_TOKENS)
            for response_chunk in st.session_state.assistant.gerar_resposta_streaming(
                prompt, contexto, max_tokens
            ):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()