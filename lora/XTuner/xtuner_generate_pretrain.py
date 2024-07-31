import json

# 设置用户的名字
# name = 'click同志'

# 设置需要重复添加的数据次数
n =  1000

# 初始化数据
data = [
    {"text": "书生·浦语大模型实战营第三期是上海人工智能实验室推出的书生·浦语大模型实战营系列活动的第三批次，将于2024年7月正式进行。"},   
]

# 通过循环，将初始化的对话数据重复添加到data列表中
for i in range(n):
    data.append(data[0])

# 将data列表中的数据写入到'datas/assistant.json'文件中
with open('datas/pretrain.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(data, f, ensure_ascii=False, indent=4)

