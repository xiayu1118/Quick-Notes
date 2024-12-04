import base64
import os
import json

import PyPDF2
import pdfplumber
from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import camelot
import tabula
from PIL import Image
import io

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import shutil
import erniebot
import numpy as np
from docx import Document
from unstructured_pytesseract import pytesseract
from zhipuai import ZhipuAI
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection, utility,db
)
from flask_cors import CORS
import jieba
from docx import Document
import erniebot
import re
from tqdm import tqdm
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})
app.config['JSON_AS_ASCII'] = False
#第一步把上传的文本进行识别
client = ZhipuAI(api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb")
api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
def create_database(names):
    db.create_database(names)
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
def using_database(names):
    db.using_database(names)
#embedding接口
def emb_text(text):
    """
    使用 ZhipuAI 的嵌入服务对文本进行嵌入。

    Args:
    - text (str): 要嵌入的文本。

    Returns:
    - np.array: 截断后的嵌入向量。
    """
    response = client.embeddings.create(
        model="embedding-2",
        input=[text],
    )
    embedding = response.data[0].embedding
    truncated_embedding = embedding[:256]  # 截断到256维
    return truncated_embedding
#连接milvus服务器接口
def connect_milvus(host, port, username="", password=""):
    """
    连接到 Milvus 服务器。

    Args:
    - host (str): Milvus 服务器的 IP 地址或主机名。
    - port (int): Milvus 服务器的端口号。
    - username (str, optional): 认证用户名。
    - password (str, optional): 认证密码。
    """
    print("开始连接 Milvus")
    connections.connect("default", host=host, port=port, user=username, password=password)
    print("连接到 Milvus 成功")
#在milvus中创建集合接口
def create_collection(collection_name, dim):
    """
    在 Milvus 中创建新的集合。

    Args:
    - collection_name (str): 集合的名称。
    - dim (int): 向量嵌入的维度。

    Returns:
    - Collection: 创建的 Milvus 集合对象。
    """
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="aaa", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    ]
    schema = CollectionSchema(fields, collection_name)

    print(f"创建集合 `{collection_name}`")
    coll = Collection(collection_name, schema, consistency_level="Bounded", shards_num=1)
    return coll
#在milvus中插入数据接口
def insert_data(collection, texts):
    """
    向指定的 Milvus 集合中插入数据。

    Args:
    - collection (Collection): Milvus 集合对象。
    - texts (list): 要插入到集合中的文本列表。
    """
    truncated_texts = [text[:256] for text in texts]  # 截断文本
    embeddings = [emb_text(text) for text in truncated_texts]  # 生成嵌入向量
    ids = list(range(len(texts)))  # 生成 ID

    data = [
        ids,
        embeddings,
        truncated_texts
    ]

    print("开始插入实体")
    insert_result = collection.insert(data)
    print("开始刷新集合")
    collection.flush()
    print("集合刷新完成")
#在milvus中创建索引接口
def create_index(collection):
    """
    在 Milvus 集合的向量字段上创建索引。

    Args:
    - collection (Collection): Milvus 集合对象。
    """
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    print("创建索引")
    collection.create_index(field_name="aaa", index_params=index_params)
    print("索引创建完成")
#删除milvus集合接口
def delete_collection(collection):
    """
    删除指定的 Milvus 集合。

    Args:
    - collection (Collection): Milvus 集合对象。
    """
    print("释放集合")
    collection.release()
    print("集合已释放")

    if collection.has_index():
        print("索引存在，正在删除")
        collection.drop_index()

    print(f"删除集合 `{collection.name}`")
    utility.drop_collection(collection.name)
    print("集合已删除")
#milvus向量检索接口
def search(collection, query_text, limit=10):
    """
    在 Milvus 集合中搜索相似的向量。

    Args:
    - collection (Collection): Milvus 集合对象。
    - query_text (str): 要查询相似嵌入的文本。
    - limit (int, optional): 返回的最大结果数量。

    Returns:
    - list: 搜索结果列表。
    """
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    query_embedding = emb_text(query_text)

    print("加载集合")
    collection.load()
    print("集合加载完成")

    print("开始搜索")
    results = collection.search(
        data=[query_embedding],
        anns_field="aaa",
        param=search_params,
        limit=limit,
        expr=None,
        output_fields=["text"]
    )

    return results
#使用文心一言回答的接口
def ask_question(context, question):
    """
    使用 ErnieBot 生成基于上下文的问题回答。

    Args:
    - context (str): 上下文信息，用于生成回答。
    - question (str): 要问 ErnieBot 的问题。

    Returns:
    - str: ErnieBot 生成的回答。
    """
    prompt = f"检索结果: {context}\n问题: {question}"
    erniebot.api_type = 'aistudio'
    erniebot.access_token = '90ba6a5d0e6d72c90ba3f50f997a533659848788'

    response = erniebot.ChatCompletion.create(
        model='ernie-3.5',
        messages=[{'role': 'user', 'content': prompt}],
    )

    restext = response.get_result()
    return restext
#pdf处理
def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本、图片和表格，并将图片和表格保存到 loader 文件夹中。

    Args:
    - pdf_path (str): PDF 文件的路径。

    Returns:
    - dict: 包含从 PDF 中提取的文本、图片和表格的字典。
    """
    result = {
        "text": "",
        "images": [],
        "tables": []
    }
    # 创建 loader 文件夹
    output_dir = os.path.join(os.path.dirname(pdf_path), 'loader')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 提取文本
            text = page.extract_text()
            if text:
                result["text"] += text

            # 提取图片
            for j, img in enumerate(page.images):
                x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
                pil_image = cropped_image.original
                image_path = os.path.join(output_dir, f"image_page_{i + 1}_{j + 1}.png")
                pil_image.save(image_path)
                result["images"].append(image_path)

    # 使用 camelot 提取表格
    tables = camelot.read_pdf(pdf_path, flavor='stream')
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"table_camelot_{i + 1}.csv")
        table.df.to_csv(table_path, index=False)
        result["tables"].append(table_path)
    #result["text"]=load_pdf(pdf_path)
    return result
#图像处理
def extract_text_from_image(path):
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
        valid_extensions = (".png", ".jpg", ".jpeg", ".gif")
        if img_file.lower().endswith(valid_extensions):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return image_summaries

def extract_text_from_image1(image_path):
    """
    Generate a summary and base64 encoded string for a single image.
    image_path: Path to the image file.
    """
    # Prompt
    prompt = """您是一位助手，负责为检索目的对图像进行摘要。\n
            这些摘要将被嵌入并用于检索原始图像。\n
            请提供一个简洁的摘要，以便优化检索。"""

    valid_extensions = (".png", ".jpg", ".jpeg", ".gif")
    if image_path.lower().endswith(valid_extensions):
        base64_image = encode_image(image_path)
        image_summary = image_summarize(base64_image, prompt)
        return image_summary
    else:
        return "Invalid file type. Please provide an image file with one of the following extensions: png, jpg, jpeg, gif."

def extract_text_from_word(docx_path):
    """
    从 Word 文档中提取文本。

    Args:
    - docx_path (str): Word 文档的路径。

    Returns:
    - str: 从 Word 文档中提取的文本。
    """
    loader = UnstructuredWordDocumentLoader(docx_path, mode ="elements", strategy ="fast",)
    docs = loader.load_and_split()
    return docs
def extract_text_from_txt(txt_path):
    """
    从文本文件中提取文本。

    Args:
    - txt_path (str): 文本文件的路径。

    Returns:
    - str: 从文本文件中提取的文本。
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
def split_text(text, chunk_size=512, chunk_overlap=32):
    """
    将文本分割为适当大小的块。

    Args:
    - text (str): 要分割的文本。
    - chunk_size (int): 每块的最大字符数。
    - chunk_overlap (int): 每块之间的重叠字符数。

    Returns:
    - list: 分割后的文本块列表。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

#多文档检索
def process_files(file_path):
    """
    处理文件以从各种格式中提取文本，并对每个文本进行分词。

    Args:
    - file_path (str): 要处理的文件路径或目录路径。

    Returns:
    - list: 包含每个文本分词结果的二维数组。
    """
    texts = []

    # 如果 file_path 是一个目录，调用图片识别接口
    if os.path.isdir(file_path):
        text = extract_text_from_image(file_path)
        texts=text
    else:
        # 处理单个文件
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                for a in text:
                    seg_list = jieba.lcut(a)
                    texts.append(seg_list)
        elif ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            for a in text['text']:
                seg_list = jieba.lcut(a)
                texts.append(seg_list)
        elif ext == ".docx":
            text = extract_text_from_word(file_path)
            text=str(text)
            texts=text

    return texts

import PyPDF2
def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        aaa= extract_text_from_image1(file_path)
        aaa=aaa.content
        aaa=str(aaa)
        return aaa
    elif ext == '.docx':
        text = extract_text_from_word(file_path)
        chunks_dict = ""
        for a in text:
            chunks_dict += a.page_content
        text = chunks_dict
        return text
    elif ext == '.txt':
        return load_text(file_path)
    else:
        return "无此格式"+str(ext)


def load_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extractText()
    except Exception as e:
        print(f"Error loading PDF file: {e}")
    return text


def load_image(file_path):
    text = ""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error loading image file: {e}")
    return text


def load_word(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error loading Word file: {e}")
    return text


def load_text(file_path):
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error loading text file: {e}")
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return extract_text_from_image(file_path)
    elif ext == ".docx":
        return extract_text_from_word(file_path)
    return "无此格式"


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
    请提供一个简洁的摘要，以便优化检索；摘要文本长度少于5000字
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
        max_chars = 500
        split_texts = []
        for text in texts:
            if len(text) > max_chars:
                split_texts.extend([text[i:i + max_chars] for i in range(0, len(text), max_chars)])
            else:
                split_texts.append(text)

        # 获取每个文本的总结
        summaries = summarize_chain.batch(split_texts, {"max_concurrency": 5})
        text_summaries = []

        # 合并总结，确保总长度不超过5000字符
        total_length = 0
        for summary in summaries:
            summary_text = summary.strip()
            if total_length + len(summary_text) > 5000:
                break
            text_summaries.append(summary_text)
            total_length += len(summary_text)

    elif texts:
        # 如果没有要求总结，则直接使用文本
        total_length = 0
        for text in texts:
            if total_length + len(text) > 5000:
                break
            text_summaries.append(text)
            total_length += len(text)

    # Apply to tables if tables are provided
    if tables:
        # 获取每个表格的总结
        summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        # 合并表格总结，确保总长度不超过5000字符
        total_length = 0
        combined_table_summaries = []
        for summary in summaries:
            summary_text = summary.strip()
            if total_length + len(summary_text) > 5000:
                break
            combined_table_summaries.append(summary_text)
            total_length += len(summary_text)
    else:
        combined_table_summaries = []

    # 合并文本和表格总结，确保总长度不超过5000字符
    final_summaries = text_summaries + combined_table_summaries
    total_length = 0
    result_summaries = []
    for summary in final_summaries:
        summary_text = summary.strip()
        if total_length + len(summary_text) > 5000:
            break
        result_summaries.append(summary_text)
        total_length += len(summary_text)
    return result_summaries

def document_to_dict(doc):
    """
    Convert Document object to a dictionary.
    """
    return {
        'metadata': doc.metadata,
        'page_content': doc.page_content
    }


