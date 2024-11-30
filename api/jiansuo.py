from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from zhipuai import ZhipuAI
from langchain_community.vectorstores import Milvus
from langchain.llms import Xinference
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough

# 替换成任何你想要进行问答的txt文件，并指定编码为utf-8
loader = TextLoader(r"C:\Users\lenovo\Desktop\12.txt", encoding="utf-8")
client = ZhipuAI(api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class CustomEmbeddings:
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        """
        使用 ZhipuAI 的嵌入服务对文档进行嵌入。
        """
        response = self.client.embeddings.create(
            model="embedding-2",
            input=texts,
        )
        embeddings = [item.embedding[:256] for item in response.data]
        return embeddings

    def embed_query(self, query):
        """
        使用 ZhipuAI 的嵌入服务对查询进行嵌入。
        """
        response = self.client.embeddings.create(
            model="embedding-2",
            input=[query],
        )
        embedding = response.data[0].embedding[:256]
        return embedding


try:
    documents = loader.load()
except Exception as e:
    print(f"加载文件时出错: {e}")
    documents = None

if documents:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    # 创建自定义嵌入实例
    custom_embeddings = CustomEmbeddings(client)

    vector_db = Milvus.from_documents(
        docs,
        custom_embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )
    retriever = vector_db.as_retriever()
    query = "步骤是什么"
    prompt = hub.pull("rlm/rag-prompt")
    retrieved_docs = vector_db.similarity_search(query, k=10)
    formatted_docs = format_docs(retrieved_docs)
    print(formatted_docs)
    # 使用相同的 Xinference 实例
    llm = Xinference(
        server_url="http://10.255.198.65:9997/",
        model_uid='attap'  # replace model_uid with the model UID return from launching the model
    )
    print(llm)


    class DocsRetriever:
        def __init__(self, retriever, docs):
            self.retriever = retriever
            self.docs = docs

        def __call__(self, inputs):
            return {"context": self.docs, "question": inputs["question"]}


    docs_retriever = DocsRetriever(retriever, formatted_docs)

    rag_chain = (
            {"context": docs_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    result = rag_chain.invoke({"question": "它的步骤是什么？"})
    print(1)
    print(result)

else:
    print("文件加载失败，无法继续处理。")
