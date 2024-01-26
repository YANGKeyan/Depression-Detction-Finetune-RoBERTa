import pandas as pd
from sklearn.utils import shuffle

#定义标签列表
labels = ['severe', 'moderate', 'not depression']

#定义文本列名称字典
text_column_names = {
    "dev": "Text data",
    "train": "Text_data",
    "test": "text data"
}

pid_column_names = {
    "dev": "PID",
    "train": "PID",
    "test": "Pid"
}

#如果需要打乱数据，则可以将use_shuffle参数设置为True。
#如果不需要标签列，则可以将without_label参数设置为True。
def get_data(data_split, use_shuffle=False, without_label=False): # 读取原始tsv转换为dataframe
    df = pd.read_csv(f'../data/original_dataset/{data_split}.tsv', sep='\t', header=0)

    pid_column = pid_column_names.get(data_split) # 获取PID列名称
    text_column = text_column_names.get(data_split) # 获取文本列名称
    label_column = "Label" # 定义标签列名称
    if without_label: # 如果不需要标签列
        df = df[[pid_column, text_column]] # 只保留PID和文本列
        df.columns = ["pid", "text"] # 重命名列
    else: #如果需要标签列
        # 将标签列转换为数字标签
        df[label_column] = df[label_column].transform(lambda label: labels.index(label))
        # 重命名列
        df.columns = ["pid", "text", "labels"]
    if use_shuffle: # 如果需要打乱顺序
        return shuffle(df) # 返回打乱后数据
    return df # 返回数据

# 定义读取预处理后的数据的函数
'''def get_preprocessed_data(data_split, use_shuffle=False):
    # 读取预处理后csv文件
    df = pd.read_csv(f'../data/preprocessed_dataset/{data_split}.csv', header=0, lineterminator='\n')
    if use_shuffle: # 打乱
        return shuffle(df)
    return df'''

def get_preprocessed_data(data_split, use_shuffle=False):
    # 读取预处理后csv文件
    try:
        df = pd.read_csv(f'../data/preprocessed_dataset/{data_split}.csv', header=0, lineterminator='\n')
        print(f"数据集中有{len(df)}行数据。")
        print("文件读取成功！")
    except:
        print("文件读取失败！")
        return None
    if use_shuffle: # 打乱
        df = shuffle(df)
        print("数据打乱成功！")
    return df