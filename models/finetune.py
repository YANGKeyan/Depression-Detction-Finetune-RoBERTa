# 使用预训练的transformer模型进行文本分类的微调
# 通过微调，模型可以更好地适应特定任务的数据集，从而提高模型的性能
import argparse # 解析命令行参数
import os
from metrics import macro_f1_score
from utils import find_best_checkpoint, clean
from transformers_models import MODEL_CLASSES
from simpletransformers.classification import ClassificationModel # 加载预训练模型
from dataset.utils import get_preprocessed_data, labels #从预处理的数据集中获取训练和评估数据
from models_list import get_models
from config import get_fine_tuning_args, global_config


def fine_tune():
    print(f'Fine-tuning\t{model_info.description()}')
    train_data = get_preprocessed_data("train", use_shuffle=True)# 获取预处理后的训练数据
    eval_data = get_preprocessed_data("dev")# 获取预处理后的评估数据

    model = ClassificationModel( # 加载预训练模型
        model_info.model_type,
        model_info.get_model_path(),
        num_labels=len(labels),
        args=model_args,
    )
    # 设置dropout
    set_dropout(model)
    # 训练模型
    model.train_model(train_data, eval_df=eval_data, f1=macro_f1_score)

    best_checkpoint = find_best_checkpoint(model_args.output_dir) # 找到最佳的checkpoint
    print(f"Best checkpoint: {best_checkpoint}") 
    clean(model_args.output_dir, best_checkpoint) # 并删除其他checkpoint


def set_dropout(model: ClassificationModel):
    config = model.config
    # 设置attention dropout
    config.attention_probs_dropout_prob = dropout.att_dropout
    # hidden_dropout
    config.hidden_dropout_prob = dropout.h_dropout
    # classifier_dropout
    config.classifier_dropout = dropout.c_dropout

    print(f"Attention dropout: {config.attention_probs_dropout_prob}")
    print(f"Hidden dropout: {config.hidden_dropout_prob}")
    print(f"Classifier dropout: {config.classifier_dropout}")


    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_info.model_type]
    model.model = model_class.from_pretrained(model_info.get_model_path(), config=config)


if __name__ == "__main__": # 解析命令行参数并循环遍历所有模型和版本，然后调用fine_tune()函数来微调每个模型
    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--models", type=str, default="basic")
    # 解析命令行参数
    args = parser.parse_args()

    # 循环遍历所有模型和版本
    for model_info in get_models(args.models):
        for version in range(global_config.runs):
            model_info.model_version = f'v{version + 1}'
            # 获取微调参数
            model_args, dropout = get_fine_tuning_args(model_info)

            # 调用finetune来微调每个模型
            fine_tune()
