import re
import json
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb")  # 填写您自己的APIKey


def remove_quotes_from_fields(json_data):
    # 定义需要去掉引号的字段名
    fields = ['data', 'text', 'uid', 'children']

    for field in fields:
        # 使用正则表达式去掉字段名的引号
        json_data = re.sub(rf'"{field}"\s*:', f'{field}:', json_data)

    return json_data


def mindmap_changess(description):
    # 构建请求的内容
    askcont = """
    帮我把下列描述生成为如下的mindmap-json数据结构，它是一个树状的父子结构，记住，你只要生成规范的数据结构返回，不能有任何的其他介绍或者描述这个数据结构的语句。
    以下是示例的mindmap-json数据结构：
    {
    data: {
        text: '',
        uid: '1'
    },
    children: [
        {
            data: {
                text: '',
                uid: '2'
            },
            children: []
        },
        {
            data: {
                text: '',
                uid: '3'
            },
            children: []
        }
    ]
    }

    接下来是描述的话：
    """
    askcont += description
    askcont += ",子结构的内容应该详细点,请注意对于这个数据格式模版中的data,children,text,uid,我不需要双引号,我不需要双引号,我不需要双引号,重要的事情说三遍，并且我只需要返回上述格式的内容即可，不需要返回其他内容"
    # 设置ErnieBot的API类型和访问令牌
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": askcont},
        ],
    )
    # 获取返回结果
    restext = response.choices[0].message.content
    # print(f"Response from ErnieBot:\n{restext}")

    # 过滤返回结果，提取有效的JSON数据结构
    # json_data = extract_json_content(restext)
    # if json_data is None:
    #     raise ValueError("No valid JSON structure found in the response.")

    # # 去掉字段名的引号
    # cleaned_json_data = remove_quotes_from_fields(json_data)

    return restext


def extract_json_content(response):
    start_marker = "```json"
    end_marker = "```"

    start_index = response.find(start_marker)
    end_index = response.find(end_marker, start_index + len(start_marker))

    if start_index == -1 or end_index == -1:
        return None  # Markers not found

    start_index += len(start_marker)
    json_content = response[start_index:end_index].strip()

    return json_content