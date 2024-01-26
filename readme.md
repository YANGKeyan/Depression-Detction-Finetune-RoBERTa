##环境 版本

Ubuntu 18.04

Tesla V100

python == 3.8

transformers == 4.13.0

simpletransformers == 0.63

pandas == 1.2.5

scikit-learn == 0.23.1

tqdm == 4.62.3

##本工程任务

基于simpletransformer对roberta模型进行预训练与微调

任务为识别英文社交媒体文本中的抑郁症迹象，分为三类：moderate、severe、not depression

##数据集预处理

python preprocessed.py

##运行预训练

python deproberta.py

##运行微调
python fintune.py

##运行分类
python predict.py
