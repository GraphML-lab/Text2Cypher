import os
import warnings
import requests
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from typing import List
import pickle
from langchain.storage import EncoderBackedStore, LocalFileStore
from langchain.text_splitter import TextSplitter
from langchain.retrievers import ParentDocumentRetriever

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning
)
## --- 1. Embedding Model Config ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
)
if not os.path.exists("/data/home/yilin/projects/test/data"):
    os.makedirs("/data/home/yilin/projects/test/data")
## --- 2. Database and Store Paths ---
URI = "/data/home/yilin/projects/test/data/milvus.db"
COLLECTION_NAME = "Cypher_Docs"


## --- 3. Custom Splitter ---
class DescriptionExtractor(TextSplitter):
    """
    A custom splitter whose purpose is to extract the child document text for searching,
    which includes the topic and description, from the parent document's HTML content.
    """

    def __init__(self, **kwargs):
        super().__init__(chunk_size=1, chunk_overlap=0, **kwargs)

    def split_text(self, text: str) -> List[str]:
        # 'text' is the page_content of a parent document, which is the full HTML of an exampleblock
        soup = BeautifulSoup(text, "lxml")

        desc_div = soup.find("div", class_="description")
        description = desc_div.get_text(separator=" ", strip=True)
        # Extract the injected header
        header_tag = soup.find("h3")
        header_text = header_tag.get_text(strip=True)

        final_content = f"{header_text}:{description}"
        return [final_content]  # Return a list containing the child document's content


## --- 4. Function to Load Parent Documents (with header injection) ---
def get_parent_html_chunks(url: str) -> List[Document]:
    """
    Loads the webpage, finds each exampleblock, "injects" the nearest preceding header
    into its content, and then returns this modified HTML as the parent document.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    full_page_soup = BeautifulSoup(response.content, "lxml")
    blocks = full_page_soup.find_all("div", class_="exampleblock")

    parent_documents = []
    for block in blocks:
        try:
            code = block.find("div", class_="listingblock").get_text(
                separator=" ", strip=True
            )
            description = block.find("div", class_="description").get_text(
                separator=" ", strip=True
            )
        except:
            continue
        else:
            header = block.find_previous("h3")
            block.insert(0, header)

            doc = Document(page_content=str(block), metadata={"source": url})
            parent_documents.append(doc)

    print(f"Successfully loaded and modified {len(parent_documents)} parent documents.")
    return parent_documents


# --- 5. Setup Retrievers and Stores ---
vector_store = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": URI},
    collection_name=COLLECTION_NAME,
    index_params={"index_type": "FLAT", "metric_type": "IP"},
    auto_id=True,
)
store_path = "/data/home/yilin/projects/test/data/docstore"
if not os.path.exists(store_path):
    os.makedirs(store_path)
base_store = LocalFileStore(store_path)

encoder_backed_store = EncoderBackedStore(
    base_store,
    key_encoder=lambda key: str(key),
    value_serializer=pickle.dumps,
    value_deserializer=pickle.loads,
)

child_splitter = DescriptionExtractor()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=encoder_backed_store,
    child_splitter=child_splitter,
    search_kwargs={"k": 3},
)

# --- 6. Populate Stores if Empty ---
if not os.path.exists("/data/home/yilin/projects/test/data/populated.txt"):
    parent_docs = get_parent_html_chunks(
        "https://neo4j.com/docs/cypher-cheat-sheet/5/neo4j-community/"
    )
    if parent_docs:
        doc_ids = [str(i) for i in range(len(parent_docs))]

        parent_retriever.add_documents(parent_docs, ids=doc_ids, add_to_docstore=True)
        with open("/data/home/yilin/projects/test/data/populated.txt", "w") as f:
            f.write("done")
    else:
        print("Failed to load documents. Exiting.")
        exit()


## --- 7. Final Retrieval and Debugging Function ---
def retrieve(question: str) -> List[str]:
    retrieved_parent_docs = parent_retriever.invoke(question)
    output = []
    for i, doc in enumerate(retrieved_parent_docs):
        soup = BeautifulSoup(doc.page_content, "lxml")
        # For readability, we print the plain text content of the parent document
        code = soup.find("div", class_="listingblock").get_text(
            separator=" ", strip=True
        )
        description = soup.find("div", class_="description").get_text(
            separator=" ", strip=True
        )
        output.append(f"{description.removesuffix('.')}:\n{code};")
    return output


if __name__ == "__main__":
    result = retrieve("To return all nodes, relationships and paths found in a query")
    for i in result:
        print(i)
        print("------------")
