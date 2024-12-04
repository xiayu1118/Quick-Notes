import time
from flask import Blueprint, jsonify
import erniebot
import pandas as pd
import re
import jieba
import difflib
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings as LangChainEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Milvus
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
import uuid
from typing import List
from langchain.schema import BaseRetriever
import jwt
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import jieba
from gensim.models import Word2Vec
keyword1 = {
    "750、500kV输变电工程可研设计一体化",
    "220、330kV输变电工程可研设计一体化",
    "110（66）kV输变电工程可研设计一体化",
    "35kV输变电工程可研设计一体化",
    "750、500kV输变电工程勘察设计",
    "220、330kV输变电工程勘察设计",
    "110（66）kV输变电工程勘察设计",
    "35kV输变电工程勘察设计",
    "35kV输变电工程施工",
    "110（66）kV输变电工程施工",
    "220kV输变电工程施工",
    "330kV输变电工程施工",
    "750kV、500kV输变电工程施工",
    "750、500kV输变电工程监理",
    "220、330kV输变电工程监理",
    "110（66）kV输变电工程监理",
    "35kV输变电工程监理",
    "安防设施设备运维服务",
    "消防设施设备运维服务",
    "应急指挥设备运维服务",
    "视频监控设备运维服务",
    "消防系统检测服务",
    "配网、业扩工程可研、勘察设计、可研设计一体化-技改检修工程勘察设计",
    "技改、修理工程可研、勘察设计、可研设计一体化-技改检修工程勘察设计",
    "配网、业扩工程施工-技改检修工程施工",
    "技改、修理工程施工，带电作业施工-技改检修工程施工",
    "配网、业扩工程监理-技改检修工程施工监理",
    "技改、修理工程监理-技改检修工程施工监理",
    "隧道结构检测，电力管廊、GIL管廊隧道结构检测评估-技改检修工程项目管理",
    "通道数据采集运维-电网设备设施委托运维服务",
    "通道辅助巡视值守-电网设备设施委托运维服务",
    "配电自动化终端运维修理-电网设备设施委托运维服务",
    "配电中低压网格化运维（10千伏及以下）-电网设备设施委托运维服务",
    "配电中低压网格化运维（20千伏）-电网设备设施委托运维服务",
    "无人机巡检-电网设备设施委托运维服务",
    "变电站辅助设备修理运维-电网设备设施委托运维服务",
    "带电水冲洗服务-电网设备设施委托运维服务",
    "机器人带电绝缘喷涂服务-电网设备设施委托运维服务",
    "设备返厂大修维护-电网设备设施委托运维服务",
    "直升机作业服务",
    "充换电站工程可研设计一体化-营销工程勘察设计",
    "计量、采集装置建设与改造工程-勘察设计营销工程勘察设计",
    "充换电站工程施工，计量、采集装置建设与改造工程施工-营销工程施工",
    "智能互动终端建设与改造施工-营销工程施工",
    "充换电站工程监理，计量、采集装置建设与改造工程-营销工程施工监理",
    "高低压采集运维-营销运维服务",
    "智能电能表维修-营销运维服务",
    "智能互动终端运维-营销运维服务",
    "充换电站设施运维-营销运维服务",
    "营业厅设备运维-营销运维服务",
    "科技技术服务-科技服务",
    "科技项目-科技服务",
    "电网环境影响评价-环评水保服务",
    "水土保持方案编制、监测环评-水保服务",
    "环境监测-环评水保服务",
    "职业危害因素监测与评价-环评水保服务",
    "沉降观测-电网工程咨询服务",
    "工程质量检测-电网工程咨询服务",
    "基坑监测-电网工程咨询服务",
    "大件运输-电网工程咨询服务",
    "工程结算审核-电网工程咨询服务",
    "规划选址评估-电网工程咨询服务",
    "节能评估-电网工程咨询服务",
    "跨铁路、航道等安全性评价-电网工程咨询服务",
    "社会稳定风险评估-电网工程咨询服务",
    "项目后评价-电网工程咨询服务",
    "技术支持服务-信息系统安全测评与加固-信息技术服务",
    "技术支持服务-信息系统客户支持服务-信息技术服务",
    "技术支持服务-数据治理服务-信息技术服务",
    "设备维保服务-信息技术服务",
    "信息系统开发-信息技术服务",
    "信息系统实施-信息技术服务",
    "仓储服务-物资仓储服务",
    "废旧物资评估-物资仓储服务",
    "运输服务-物资仓储服务",
    "水路运输服务-物资仓储服务",
    "装卸搬运-物资仓储服务",
    "废旧物资处置-物资仓储服务",
    "仓储设备运维-物资质量管理服务",
    "设备监造-物资质量管理服务",
    "设备检验-物资质量管理服务",
    "土建工程设计/可研设计一体化、智能化设计/可研设计一体化、装修装饰设计/可研设计一体化-小型基建工程勘察设计",
    "土建施工-小型基建工程施工",
    "智能化施工-小型基建工程施工",
    "装修装饰施工-小型基建工程施工",
    "土建、智能化、装饰装修、总承包工程监理-小型基建工程监理",
    "工程检测-小型基建工程咨询服务",
    "设计、施工-小型基建工程总承包",
    "房屋维修-房屋维修设计-后勤运维服务",
    "房屋维修-房屋维修施工-后勤运维服务",
    "房屋维修-房屋维修监理-后勤运维服务",
    "车辆维修-车辆维修服务-后勤运维服务",
    "车辆维修-车辆装潢及换轮胎服务-后勤运维服务",
    "安保服务-后勤运维服务",
    "特种设备运维-后勤运维服务",
    "船舶维修-后勤运维服务",
    "绿化美化服务-物业及绿化服务",
    "食堂服务-物业及绿化服务",
    "卫生保洁-物业及绿化服务",
    "物业管理-物业及绿化服务",
    "车辆租赁-物业及绿化服务",
    "船舶租赁-物业及绿化服务",
    "印刷服务（出版物、包装装潢印刷品等经营性排版、制版、印刷服务）-物业及绿化服务",
    "文印服务-文印服务（资料打印、复印等文印服务）-物业及绿化服务",
    "通信工程施工-调度工程专业",
    "调度设备运维修理-调度运维修理服务专业",
    "通信设备运维-调度运维修理服务专业",
    "无线专网设备运维-调度运维修理服务专业",
    "财产保险服务-保险服务",
    "其他保险服务-保险服务",
    "中介服务-代理服务-税务服务-财务服务",
    "中介服务-代理服务-会计服务-财务服务",
    "中介服务-法律服务",
    "培训服务",
    "人力资源服务",
    "劳务派遣服务",
    "中介服务-审计服务-工程结算审计",
    "中介服务-审计服务-工程决算审计",
    "广告宣传服务",
    "管理咨询"
}
keyword2 = {
    "采购类别",
    "人员要求",
    "业绩要求",
    "资质要求"
}
bp = Blueprint('excel_match', __name__, url_prefix='/excel_match')
app_routes=r"C:\Users\lenovo\Desktop\others\EditEnd\apps\excel_data"
def find_top_n_matches(input_word, keywords, top_n=5, cutoff=0.4):
    # 使用 difflib 的 get_close_matches 查找最接近的匹配
    matches = difflib.get_close_matches(input_word, keywords, n=top_n, cutoff=cutoff)
    return matches


# 从较长句子中提取关键短语
def extract_relevant_phrases(user_input, window_size=4):
    words = jieba.lcut(user_input)
    phrases = [''.join(words[i:i + window_size]) for i in range(len(words) - window_size + 1)]
    return phrases
# 第二次从候选词中匹配最精确的关键词，若无精确匹配则返回最相近的
def match_from_candidates(input_word, candidates):
    # 先检查是否有精确匹配
    if input_word in candidates:
        return input_word
    # 若没有精确匹配，则返回最相近的一个
    matches = find_top_n_matches(input_word, candidates, top_n=1)
    if len(matches) != 0:
        return matches[0]
    else:
        if len(candidates)!=0:
          return candidates[0]
        else:
            return None


# 将输入转换为 'keyword1 的 keyword2' 形式
def convert_to_keyword_format(user_input):
    # 滑动窗口提取短语
    phrases = extract_relevant_phrases(user_input)

    # 匹配 keyword1 和 keyword2
    candidate_keyword1 = []
    candidate_keyword2 = []

    for phrase in phrases:
        if not candidate_keyword1:
            candidate_keyword1 = find_top_n_matches(phrase, keyword1, top_n=5)
        if not candidate_keyword2:
            candidate_keyword2 = find_top_n_matches(phrase, keyword2, top_n=5)
        if candidate_keyword1 and candidate_keyword2:
            break
    # 第二次匹配：在候选项中进行更精确的匹配
    final_keyword1 = match_from_candidates(user_input, candidate_keyword1)
    final_keyword2 = match_from_candidates(user_input, candidate_keyword2)

    # 如果都匹配到了，返回 'keyword1 的 keyword2' 形式
    if final_keyword1 and final_keyword2:
        return f"{final_keyword1} 的 {final_keyword2}"
    else:
        return "未能匹配到合适的关键词"
# 自定义 Milvus 检索器
class MilvusRetriever(BaseRetriever):
    def __init__(self, collection_name):
        self.collection = Collection(collection_name)

    def _get_relevant_documents(self, query_embedding, top_k=5):
        # 搜索 Milvus 中最相关的文档
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        search_results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "category", "qualification", "performance", "personnel", "text", "entry_id"]  # 添加 `text` 字段到输出字段中
        )
        # 提取返回的文档
        docs = []
        for result in search_results[0]:
            doc = {
                "id": result.entity.get("id"),
                "category": result.entity.get("category"),
                "qualification": result.entity.get("qualification"),
                "performance": result.entity.get("performance"),
                "personnel": result.entity.get("personnel"),
                "text": result.entity.get("text"),
                "entry_id": result.entity.get("entry_id")
            }
            docs.append(doc)
        return docs
def clean_input(user_input):
    # 去掉无关词，比如标点符号
    cleaned = re.sub(r'[^\w\s]', '', user_input)
    return cleaned

def create_milvus_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="qualification", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="performance", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="personnel", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="entry_id", dtype=DataType.VARCHAR, max_length=10000, is_primary=True)
    ]

    schema = CollectionSchema(fields=fields, description='search text')
    collection = Collection(name=collection_name, schema=schema)

    # 为 vector 字段构建索引
    index_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='vector', index_params=index_params, index_name="idx_em")
    return collection
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

# 定义 SentenceTransformer 嵌入类
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str):
        return self.model.encode([query])[0].tolist()

# 初始化嵌入模型
embeddings = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

@bp.route('/insert_data', methods=['POST', 'GET'])
def insert_data():
    try:
        # 连接 Milvus
        connections.connect(alias="default", host='127.0.0.1', port='19530')
        exc = request.files.get("files")

        if exc is None:
            return jsonify({'success': 'false', 'message': '未收到上传的文件'}), 400

        docname = "资质业绩-工程及服务"

        # 确保保存文件的路径存在
        if not os.path.exists(app_routes):
            os.makedirs(app_routes)
        os.makedirs(app_routes, exist_ok=True)

        # 保存文件
        file_path = os.path.join(app_routes, docname)
        if os.path.exists(file_path):
            return jsonify({'success':'false','message':'已经存在对应的文件了，无法继续导入'}), 400
        exc.save(file_path)

        # 读取 Excel 文件内容
        data = pd.read_excel(file_path, header=0)
        data = data.loc[:, ['序号', '采购类别', '资质要求', '业绩要求', '人员要求']]  # 按需选择的列

        # 重命名列名，以符合 Milvus 的命名规则
        data = data.rename(columns={
            '序号': 'id',
            '采购类别': 'category',
            '资质要求': 'qualification',
            '业绩要求': 'performance',
            '人员要求': 'personnel'
        })

        # 确保所有列都是字符串类型
        data = data.astype(str)

        # 合并数据为 `text` 字段
        data['text'] = data.apply(lambda row: ' '.join(row.values), axis=1)

        # 将数据转换为字典格式
        results = data.to_dict('records')

        # 提取向量并更新数据
        expanded_results = []
        for i in range(len(results)):
            row = results[i]
            categories = ['category', 'qualification', 'performance', 'personnel']  # 需要分别向量化的字段
            for category in categories:
                # 获取对应类别的文本
                text_content = row[category]
                # 生成向量
                vector = embeddings.embed_documents([text_content])[0]
                # 创建一个新的记录
                new_entry = {
                    'id': row['id'],  # 保持原来的 id
                    'category': row['category'],  # 保持原来的 采购类别
                    'qualification': row['qualification'],  # 保持原来的 资质要求
                    'performance': row['performance'],  # 保持原来的 业绩要求
                    'personnel': row['personnel'],  # 保持原来的 人员要求
                    'text': text_content,  # 单独的部分作为 text
                    'vector': vector,  # 该部分的向量
                    'date': '2024-03',  # 保持原来的日期
                    'entry_id': str(uuid.uuid4())  # 为每个新条目生成唯一的 entry_id
                }
                expanded_results.append(new_entry)

        # 将结果转换为 DataFrame
        df_expanded = pd.DataFrame(expanded_results)

        # 检查 Milvus 中是否已有集合，若没有则创建
        collection_name = "paper"
        if not utility.has_collection(collection_name):
            collection = create_milvus_collection(collection_name, 384)  # 384 是嵌入向量的维度
        else:
           return  jsonify({'success': 'false','message':'数据插入失败，已存在相关数据'})

        # 分批插入数据，防止一次插入过多导致内存溢出
        # 插入数据到 Milvus
        batch_size = 10000
        for i in range(0, len(df_expanded), batch_size):
            batch = df_expanded.iloc[i:i + batch_size].to_dict(orient='records')
            collection.insert(batch)

        # 加载集合并检查实体数量
        collection.load()
        num_entities = collection.num_entities

        # 返回成功响应
        return jsonify({'success': 'true', 'data': f"成功插入数据：{num_entities} 条"}), 200

    except Exception as e:
        return jsonify({'success': 'false', 'message': str(e)}), 500

@bp.route('/match_query', methods=['POST','GET'])
def match_query():
    connections.connect(alias="default", host='127.0.0.1', port='19530')
    collection_name = "paper"
    collection=Collection(collection_name)
    question = request.form.get("question")
    question=convert_to_keyword_format(question)
    top_k=5
    question_embedding = embeddings.embed_query(question)
    collection.load()
    # 检索最相似的内容
    search_params = {"metric_type": "L2", "params": {"nprobe": 50}}
    #修改nprobe字段，字段值越大则匹配的精准度会更高
    results = []
    for field in ["category", "qualification", "performance", "personnel"]:
        result = collection.search(
            data=[question_embedding],
            anns_field='vector',
            param=search_params,
            limit=top_k,
            output_fields=["category", "qualification", "performance", "personnel"]
        )
        results.append(result)
    query_result = result
    prompt = ("""
    你是一个招投标领域的专家，用户现在向你提问招标方面的相关问题。请根据你的专家知识库中的内容进行回答。
    当前知识库中检索到的内容如下：
    {}
    用户的问题如下：
    {}
    请注意，你的回答必须严格依据检索到的内容，不能包含任何与检索到的原文不相关的信息。如果检索没有结果，请如实说明，并告诉用户你没有检索到对应的结果。
    用户可能提问的内容与检索到的结果的对应关系：
    用户提问的内容:检索结果的属性列名称
    序号: id
    采购类别: category
    资质要求: qualification
    业绩要求: performance
    人员要求: personnel
    如果根据用户的问题无法匹配到合适的条目，此时请你将用户可能想问的问题以目录的形式罗列出来，但如果可以比较完美地匹配，那么只需要回答你检索到的包容即可
    """.format(query_result, question))
    erniebot.api_type = 'aistudio'
    erniebot.access_token = '90ba6a5d0e6d72c90ba3f50f997a533659848788'
    response = erniebot.ChatCompletion.create(
        model='ernie-bot',
        messages=[{'role': 'user', 'content': prompt}],
    )
    restext = response.get_result()
    print(restext)
    return jsonify(restext), 200


@bp.route('/delete_excel', methods=['GET', 'POST'])
def delete_excel():
    connections.connect(alias="default", host='127.0.0.1', port='19530')
    collection_name = "paper"
    docname = "资质业绩-工程及服务"
    file_path = os.path.join(app_routes,  docname)
    if os.path.exists(file_path):
        os.remove(file_path)
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    return jsonify({'success': 'true', 'message': '数据删除成功'}), 200

@bp.route('/questions_changed', methods=['GET', 'POST'])
def questions_changed():
    user_input=request.form.get('question')
    result=convert_to_keyword_format(user_input)
    return jsonify({'success': 'true', 'data': result}), 200














