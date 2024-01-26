import os
import torch
import torch.nn as nn
from config import get_lm_pretraining_args # 导入getlmpretrainingargs函数，用于获取预训练参数
from simpletransformers.language_modeling import LanguageModelingModel # 导入simpletransformers库中的语言模型类
import logging # 导入logging模块，用于记录日志
import pandas as pd


# 配置日志记录器
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


class DepRoBERTa:

    def __init__(self, 
                 model_type="roberta",
                 model_name="roberta-large",
                 model_version="v1", # 模型版本
                 pretrain_model=None, # 预训练模型路径为空 ？
                 data_dir="../data/reddit_depression_corpora", # 数据目录
                 data_name='reddit_depression_corpora', # 数据名称
               
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.model_version = model_version
        self.pretrain_model = pretrain_model
        self.data_dir = data_dir
        self.data_name = data_name
        self.lm_args = get_lm_pretraining_args() # 获取与训练参数
        self.lm_args.output_dir = f"trained_models/deproberta_{model_version}" #输出目录
        self.lm_args.cache_dir = f"trained_models/deproberta_{model_version}/cache"#缓存目录


        self.model = self._get_model() #获取模型

     

    def train(self):
        train_data, val_data = self._get_data() # 获取训练数据和验证数据
        self.model.train_model(train_data, eval_file=val_data) # 训练模型
       
    def eval(self):
        _, val_data = self._get_data()  # 获取验证数据
        self.model.eval_model(val_data) # 评估模型


    def _get_model(self): # 获取语言模型
         # 如果预训练模型路径不为空，则使用预训练模型路径，否则使用模型名称
        model_path = self.pretrain_model if self.pretrain_model is not None else self.model_name
        # 返回语言模型
        return LanguageModelingModel(self.model_type, model_path, args=self.lm_args)

    def _get_data(self): # 获取数据路径
        # 获取训练数据路径
        train_data = os.path.join(self.data_dir, f'{self.data_name}_train.txt')
        # 获取验证数据路径
        val_data = os.path.join(self.data_dir, f'{self.data_name}_val.txt') 
        return train_data, val_data


if __name__ == "__main__":
    deproberta = DepRoBERTa() # 创建DepRoBERTa对象
    deproberta.train()
    deproberta.eval()

