class ModelInfo:

    def __init__(self, model_type: str, # 类型
                 model_name: str, # 名称
                 model_version: str = None, # 版本
                 pretrain_model: str = None): # 预训练模型
        self.model_type = model_type
        self.model_name = model_name
        self.model_version = model_version
        self.pretrain_model = pretrain_model

    def get_model_path(self): # 返回预训练模型路径
        #如果没有预训练模型pretrain_model，返回模型名称
        return self.pretrain_model if self.pretrain_model is not None else self.model_name

    def description(self): # 模型的类型、名称和版本
        return f'{self.model_type}\t{self.model_name}\t{self.model_version}'

    def simple_name(self): # 模型名称的简单版本
        return self.model_name.replace('/', '_')


models_options = {
    'basic': [
        ModelInfo('bert', 'bert-base-cased'),
        ModelInfo('bert', 'bert-large-cased'),
        ModelInfo('roberta', 'roberta-base'),
        ModelInfo('roberta', 'roberta-large'),
        ModelInfo('xlnet', 'xlnet-base-cased'),
        ModelInfo('xlnet', 'xlnet-large-cased')
    ],
    'DepRoBERTa': [
        ModelInfo('roberta', 'rafalposwiata/deproberta-large-v1')
    ],
    'best': [
        ModelInfo('roberta', 'rafalposwiata/roberta-large-depression'),
        ModelInfo('roberta', 'rafalposwiata/deproberta-large-depression')
    ]
}


def get_models(models: str):
    return models_options.get(models)
