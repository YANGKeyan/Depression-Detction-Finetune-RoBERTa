from models.models_list import get_models  # 导入 get_models 函数，用于获取最佳模型列表
from simpletransformers.classification import ClassificationModel
from dataset.utils import get_data, labels # 导入 get_data 函数和 labels 列表，用于获取测试数据和标签
import pandas as pd
import numpy as np
from scipy.special import softmax # 导入softmax函数，用于计算模型输出的概率分布


def predict_for_single_model(model_info):
    """
    对单个模型进行预测，并生成包含预测结果的 TSV 文件。

    Args:
        model_info: 包含模型信息的 ModelInfo 对象。

    Returns:
        无。
    """
    # 输出正在使用的模型信息
    print(f'Generating predictions using {model_info.description()}')
    # 对测试数据进行分类，返回预测结果和模型输出的原始结果
    predictions, raw_outputs = predict(model_info)
    # 生成包含预测结果的 TSV 文件，模型简称，预测结果
    generate_file(model_info.simple_name(), predictions)


def predict_for_average_ensemble(best_models):
    """
    对最佳模型列表进行平均集成，并生成包含预测结果的 TSV 文件。

    Args:
        best_models: 最佳模型列表，包含多个 ModelInfo 对象。

    Returns:
        无。
    """
    # 对最佳模型列表中的第一个模型进行预测，返回预测结果和模型输出的原始结果
    _, y1 = predict(best_models[0])
    # 对模型输出的原始结果y1进行 softmax 处理，使其满足概率分布的性质，得到概率分布
    # y1的值不一定再[0,1]范围内，也不一定满足概率分布的性质（即所有值的和为1）
    # 这样处理后，y1 就可以被解释为模型对每个类别的预测概率，可以根据这些概率来进行分类预测
    y1 = softmax(y1, axis=1)
    # 对最佳模型列表中的第二个模型进行预测，返回预测结果和模型输出的原始结果
    _, y2 = predict(best_models[1])
    # 对模型输出的原始结果y2进行 softmax 处理，使其满足概率分布的性质，得到概率分布
    y2 = softmax(y2, axis=1)

    # 对两个模型的概率分布进行平均，并取最大值作为预测结果
    # np.argmax 函数用于返回沿着指定轴的最大值的索引
    # 对于每个样本，predictions 中的相应元素是两个模型输出的平均值中概率最大的那个类别的索引
    predictions = np.argmax((y1 + y2) / 2, axis=1)
    # 生成包含集成预测结果的 TSV 文件
    generate_file('Ensemble', predictions)


def predict(model_info):
    """
    对测试数据进行分类，并返回预测结果和模型输出的原始结果。

    Args:
        model_info: 包含模型信息的 ModelInfo 对象。

    Returns:
        predictions: 预测结果，包含多个整数。
        raw_outputs: 模型输出的原始结果，包含多个数组。
    """
    # 使用ClassificationModel 类创建分类模型的语句
    model = ClassificationModel(
        model_info.model_type, # 模型类型
        model_info.model_name, # 模型名称
        num_labels=len(labels), # 模型需要分类的标签数量 即数据集中的类别数
    ) # 加载预训练模型

    # 对测试数据进行分类
    predictions, raw_outputs = model.predict(list(test_data["text"].values))
    return predictions, raw_outputs


def generate_file(run_name, predictions):
    """
    生成包含预测结果的 TSV 文件。

    Args:
        run_name: 运行名称，用于生成文件名。
        predictions: 预测结果，包含多个整数。

    Returns:
        无。
    """

    result = []
    for pid, prediction in zip(list(test_data["pid"].values), predictions):
        # 将预测结果转换为标签，并添加到结果列表中
        result.append([pid, labels[prediction]])
    # 将结果列表保存为 TSV 文件
    pd.DataFrame(result, columns=["pid", "class_label"]).to_csv(f"{run_name}.tsv", sep='\t', index=False)


if __name__ == "__main__":
    # 获取测试数据
    test_data = get_data("test", without_label=True)

    # 获取最佳模型列表
    # best_models = get_models('best')
    best_models = get_models('DepRoBERTa')
    

    for model_info in best_models:
        # 对每个模型进行预测，并生成包含预测结果的 TSV 文件
        predict_for_single_model(model_info)

    # 对最佳模型列表进行平均集成，并生成包含预测结果的 TSV 文件
    #predict_for_average_ensemble(best_models)
