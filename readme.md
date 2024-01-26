## Environment
Ubuntu 18.04

Tesla V100

python == 3.8

transformers == 4.13.0

simpletransformers == 0.63

pandas == 1.2.5

scikit-learn == 0.23.1

tqdm == 4.62.3

## Task

基于simpletransformer对roberta模型进行预训练与微调

任务为识别英文社交媒体文本中的抑郁症迹象，分为三类：moderate、severe、not depression

## Dataset preprocessing

python preprocessed.py

## Pretrain

python deproberta.py

## Fintune
python fintune.py

## Classify
python predict.py
