import pandas as pd
from collections import Counter # 用于计数
from dataset.utils import get_data, labels # 导入自定义的get_data和labels函数


dev_probability = {  # 定义一个字典，表示开发集中每个标签的概率
    'severe': 0.09,
    'moderate': 0.51,
    'not depression': 0.4
}


def preprocess():
    train = get_data('train') # 获取训练集数据 读取train.csv
    statistics('train', train) # 对训练集数据进行统计

    dev = get_data('dev') # 获取开发集数据 读取dev.csv
    statistics('dev', dev) # 对开发集数据进行统计

    buckets = {l: [] for l in labels} # 定义一个字典，表示每个标签对应的数据集
    all_texts = set() # 定义一个集合，用于存储所有的文本
    for data in [train, dev]: # 遍历训练集和开发集
        for idx, row in data.iterrows(): # 遍历数据集中的每一行
            pid = row['pid']
            text = row['text']
            label = row['labels']
            if text not in all_texts: # 如果文本不在集合中
                buckets[labels[label]].append([pid, text, label]) # 将该行数据添加到对应标签的数据集中
                all_texts.add(text) # 将文本添加到集合中

    # 定义两个空列表，用于存储训练集和开发集的数据
    # 合并原始数据集重新分配
    train_dataset, dev_dataset = [], []
    # 遍历开发集中每个标签的概率
    for label, prob in dev_probability.items():
        # 计算该标签需要取出的数据量
        v = int(prob * 1000)
        # 将该标签的前v个数据添加到训练集中
        train_dataset += buckets[label][:-v]
        # 将该标签的后1000-v个数据添加到开发集中
        dev_dataset += buckets[label][-v:]

    # 将训练集数据转换为DataFrame格式
    train = pd.DataFrame(train_dataset, columns=['pid', 'text', 'labels'])
    # 输出训练集数据的统计信息
    print_stats('Train after preprocessing', train)
    # 将训练集数据保存为csv文件
    train.to_csv('../data/preprocessed_dataset/train.csv', index=False)

    dev = pd.DataFrame(dev_dataset, columns=['pid', 'text', 'labels'])
    print_stats('Dev after preprocessing', dev)
    dev.to_csv('../data/preprocessed_dataset/dev.csv', index=False)


def statistics(data_split, dataset): # 统计数据集中每个标签的数量，并输出数据集的总量
    unique_data = [] # 定义一个空列表，用于存储去重后的数据
    all_texts = set() # 定义一个集合，用于存储所有的文本
    for idx, row in dataset.iterrows(): # 遍历数据集中的每一行
        if row['text'].lower() not in all_texts: # 如果文本的小写形式不在集合中
            unique_data.append([row['pid'], row['text'], row['labels']]) # 将该行数据添加到去重后的数据列表中
            all_texts.add(row['text'].lower()) # 将文本的小写形式添加到集合中

    # 输出原始数据的统计信息
    print_stats(f'Original {data_split}', dataset)
    # 将去重后的数据转换为DataFrame格式
    df = pd.DataFrame(unique_data, columns=['pid', 'text', 'labels'])
    # 输出去重后的数据的统计信息
    print_stats(f'Original {data_split} - without duplicates', df)


# 定义一个名为print_stats的函数，用于输出数据的统计信息
def print_stats(description, dataset):
    print(description) # 输出描述信息
    counts = Counter(list(dataset['labels'].values)) # 统计每个标签的数量
    for idx, label in enumerate(labels): # 遍历所有标签
        print(f'{label}: {counts[idx]}') # 输出该标签的数量
    print(f'all: {len(dataset)}')  # 输出数据集的总数量
    print('-------------------------------------') # 输出分割线


if __name__ == '__main__':
    preprocess()
