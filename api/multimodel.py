# %%
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
# Extract elements from PDF
api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )
# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """您是一位助手，负责为检索目的对表格和文本进行摘要。
    这些摘要将被嵌入并用于检索原始文本或表格元素。
    请提供一个简洁的摘要，以便优化检索。
    表格或文本：{element} """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    llm = QianfanChatEndpoint(streaming=True,
                              model="ERNIE-Bot",
                              qianfan_ak="9SGPGPstUiPahZGqhkjFCu5m",
                              qianfan_sk="jmNNhNGiGXmuhb2rnofLRsuzf9ThUK6M")
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        # 分割文本列表，确保每个文本都不超过11200个字符
        max_chars = 11200
        split_texts = []
        for text in texts:
            if len(text) > max_chars:
                split_texts.extend([text[i:i + max_chars] for i in range(0, len(text), max_chars)])
            else:
                split_texts.append(text)
        text_summaries = summarize_chain.batch(split_texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries

import base64
import os
from zhipuai import ZhipuAI
def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def image_summarize(img_base64, prompt):
    """Make image summary"""
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model="glm-4v",  # 填写需要调用的模型名称
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            }
        ]
    )
    return response.choices[0].message
def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """您是一位助手，负责为检索目的对图像进行摘要。\n
            这些摘要将被嵌入并用于检索原始图像。\n
            请提供一个简洁的摘要，以便优化检索。"""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries
# %%
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()
def create_multi_vector_retriever(
        vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        # 确保 doc_summaries 和 doc_contents 中的每个元素都是字符串类型
        summary_docs = [
            Document(page_content=str(s), metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        doc_contents_str = [str(content) for content in doc_contents]

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents_str)))

    # Add texts, tables, and images
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    return retriever
# %%
import io
import re
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))
def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [
        {"type": "text", "text": (
            "你是一名AI科学家，负责提供基于事实的答案。\n"
            "你将会接收到文本、表格和图像（通常是图表或图形）的混合信息。\n"
            "利用这些信息来回答与用户问题相关的内容。并希望你使用中文"
            f"用户提供的问题: {data_dict['question']}\n\n"
            "文本或表格:\n"
            f"{formatted_texts}"
        )}
    ]

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

    return messages
def call_zhipuai(msgs, api_key):
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model="glm-4v",
        messages=[
            {
                "role": "user",
                "content": msgs
            }
        ]
    )
    # Extract the content from the response object
    return response.choices[0].message.content
def multi_modal_rag_chain(retriever, api_key):
    """
    Multi-modal RAG chain
    """
    # RAG pipeline
    chain = (
            {
                "context": retriever | RunnableLambda(split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(img_prompt_func)
            | RunnableLambda(lambda messages: call_zhipuai(messages, api_key))
            | StrOutputParser()
    )

    return chain




api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
#获取pdf的text，tables抽取
# File path
# fpath = r"C:\Users\lenovo\Desktop\bbbbb"
# fname = r"\dify_llms_app_stack_en.pdf"
#
# # Get elements
# raw_pdf_elements = extract_pdf_elements(fpath, fname)
#
# # Get text, tables
# texts, tables = categorize_elements(raw_pdf_elements)
#
# # Optional: Enforce a specific token size for texts
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1024, chunk_overlap=256
# )
# joined_texts = " ".join(texts)
# texts_4k_token = text_splitter.split_text(joined_texts)
#
# #获取pdf摘要
# text_summaries, table_summaries = generate_text_summaries(
#     texts_4k_token, tables, summarize_texts=True
# )
#
# #获取图片信息
# fpath = r"C:\Users\lenovo\Desktop\aaaaa"
# # Image summaries
# img_base64_list, image_summaries = generate_img_summaries(fpath)
# print("image_summaries\n")
# print(image_summaries)
# #构建向量数据库
# # The vectorstore to use to index the summaries
# vectorstore = Chroma(
#     collection_name="mm_rag_cj_blog",
#     embedding_function=SentenceTransformerEmbeddings(r"C:\Users\lenovo\Desktop\others\EditEnd\text2vec-base-chinese")
# )
# # Create retriever
# retriever_multi_vector_img = create_multi_vector_retriever(
#     vectorstore,
#     text_summaries,
#     texts,
#     table_summaries,
#     tables,
#     image_summaries,
#     img_base64_list,
# )
# #构建RAG
# # Create RAG chain
# chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img, api_key)
#
# # %%
# query = """这篇文章的主要内容是什么"""
# docs = retriever_multi_vector_img.get_relevant_documents(query, limit=1)
# docs[0]
#
# # %%
# plt_img_base64(docs[0])
#
# # %%
# image_summaries[1]
#
# # %%
# query = "李昌霖获得了什么奖"
# print(chain_multimodal_rag.invoke(query))
