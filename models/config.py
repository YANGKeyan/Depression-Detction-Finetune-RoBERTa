from simpletransformers.config.model_args import LanguageModelingArgs, ClassificationArgs
#LanguageModelingArgs库用于语言模型的训练和预测，而ClassificationArgs库用于分类模型的训练和预测。


class GlobalConfig:
    def __init__(self):
        self.dir_with_models: str = 'trained_models' #训练好的模型
        self.runs: int = 5


class Dropout:
    def __init__(self, att_dropout, h_dropout, c_dropout):
        self.att_dropout = att_dropout
        self.h_dropout = h_dropout
        self.c_dropout = c_dropout


def get_fine_tuning_args(model_info): #获取微调模型的参数的
    model_args = ClassificationArgs()
    model_args.learning_rate = 5e-6
    model_args.train_batch_size = 16
    model_args.num_train_epochs = 10
    model_args.evaluate_during_training_steps = 100
    model_args.max_seq_length = 300
    model_args.weight_decay = 0.1
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.output_dir = f'{global_config.dir_with_models}{model_info.model_name}_{model_info.model_version}'

    dropout = Dropout(0.1, 0.1, 0.1)
    return model_args, dropout


def get_lm_pretraining_args(): #获取语言模型预训练的参数的
    lm_args = LanguageModelingArgs()
    lm_args.learning_rate = 4e-5
    lm_args.train_batch_size = 50
    lm_args.eval_batch_size = 50
    lm_args.num_train_epochs = 10
    lm_args.dataset_type = "simple"
    lm_args.sliding_window = True
    lm_args.overwrite_output_dir = True
    lm_args.reprocess_input_data = True
    lm_args.evaluate_during_training = True
    lm_args.evaluate_during_training_silent = True
    lm_args.save_steps = 5000
    lm_args.evaluate_during_training_steps = 5000
    return lm_args


global_config = GlobalConfig()
