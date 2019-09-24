class Args(object):
    def __init__(self, model, out_path):
        self.bert_model = model
        self.data_dir = out_path
        self.output_dir = out_path