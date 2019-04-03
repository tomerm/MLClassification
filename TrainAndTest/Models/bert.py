import os
import numpy
import random
import torch
import datetime
from torch.utils.data import TensorDataset, DataLoader as BertDataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from tqdm import tqdm, trange
from Models.base import BaseModel
from Models.bertClassifier import BertForMultiLabelSequenceClassification, \
     Args, DataProcessor, convert_examples_to_features, getLogger, accuracy
from Data.data import composeTsv
from Utils.utils import fullPath, showTime

class BertModel(BaseModel):
    def __init__(self, Config):
        super().__init__(Config)
        self.Config = Config
        self.useProbabilities = True
        self.maxBertSeqLength = 512
        self.device = 'cpu'
        self.n_gpu = torch.cuda.device_count()
        self.model_to_save = None
        if len(Config["bertpath"]) == 0 or not os.path.isfile(fullPath(Config, "bertpath")):
            print ("Wrong path to archive with pre-trained BERT model. Stop.")
            Config["error"] = True
            return
        if len(Config["bertoutpath"]) == 0 or not os.path.isdir(fullPath(Config, "bertoutpath")):
            print("Wrong path to folder with resulting BERT files. Stop.")
            Config["error"] = True
            return
        self.args = Args(fullPath(self.Config, "bertpath"), fullPath(self.Config, "bertoutpath")) # model: pytorch_ber.gz
        self.max_seq_length = min(self.maxBertSeqLength, self.Config["maxseqlen"])
        if self.Config["runfor"] != "test":
            self.do_train = True
        if self.Config["runfor"] != "train":
            self.do_eval = True
        self.do_lower_case = False
        self.train_batch_size = min(self.trainBatch, 32)
        self.eval_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = self.epochs
        self.warmup_proportion = 0.1
        self.no_cuda = True
        self.local_rank = -1
        self.seed = 42
        self.gradient_accumulation_steps = 1
        self.keyTrain = "traindocs"
        self.keyTest = "testdocs"
        #self.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        if self.Config["runfor"] != "crossvalidation":
            self.prepareData()
        self.launchProcess()

    def prepareData(self):
        print ("Start data preparation...")
        composeTsv(self, "train")
        composeTsv(self, "test")
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if os.path.exists(self.args.output_dir) and self.do_train:
            print ("Output directory ({}) already exists and is not empty.".format(self.args.output_dir))
            print ("Its content will be deleted.")
        if self.do_train:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.processor = DataProcessor(self.Config["cats"])
        self.num_labels = len(self.Config["cats"])
        self.label_list = self.processor.get_labels()
        #self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.do_lower_case)
        self.vocabPath = os.path.dirname(self.args.bert_model) + "/vocab.txt"
        self.tokenizer = BertTokenizer(self.vocabPath)

    def createModel(self):
        self.train_examples = self.processor.get_train_examples(self.args.data_dir)
        self.num_train_optimization_steps = int(
            len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps)\
                                       * self.num_train_epochs
        model = BertForMultiLabelSequenceClassification.from_pretrained(self.args.bert_model,
                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                            self.local_rank), num_labels=self.num_labels)
        model.to(self.device)
        return model

    def trainModel(self):
        print("Start training..")
        ds = datetime.datetime.now()
        param_optimizer = [p for p in self.model.named_parameters()]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        train_features = convert_examples_to_features(
            self.train_examples, self.label_list, self.max_seq_length, self.tokenizer)
        logger = getLogger()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_examples))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = BertDataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        self.model.train()
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        de = datetime.datetime.now()
        print("Model is trained in %s" %  (showTime(ds, de)))
        if self.Config["runfor"] == "crossvalidation":
            return
        print ("Model evaluation...")
        eval_examples = self.processor.get_dev_examples(self.args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, self.label_list, self.max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = BertDataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        allLabs = None
        res = None
        initRes = True
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)
            preds = logits.sigmoid().to('cpu').numpy()
            labs = label_ids.to('cpu').numpy()
            if initRes == True:
                res = preds
                allLabs = labs
                initRes = False
            else:
                res = numpy.concatenate((res, preds))
                allLabs = numpy.concatenate((allLabs, labs))
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        eval_accuracy = eval_accuracy / nb_eval_examples
        print ("Model accuracy: %.2f"%(eval_accuracy))
        # Save a trained model
        self.model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = fullPath(self.Config, "bertoutpath", opt="name")
        torch.save(self.model_to_save.state_dict(), output_model_file)
        print ("Model is saved in %s"%(output_model_file))

    def testModel(self):
        print ("Start testing...")
        ds = datetime.datetime.now()
        if self.model_to_save == None:
            output_model_file = fullPath(self.Config, "bertoutpath", opt="name")
            model_state_dict = torch.load(output_model_file)
            model = BertForMultiLabelSequenceClassification.from_pretrained(self.args.bert_model,
                                            state_dict=model_state_dict, num_labels=self.num_labels)
            model.to(self.device)
        eval_examples = self.processor.get_dev_examples(self.args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, self.label_list, self.max_seq_length, self.tokenizer)
        #self.logger.info("  Num examples = %d", len(eval_examples))
        #self.logger.info("  Batch size = %d", self.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = BertDataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        allLabs = None
        res = None
        initRes = True
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)
            preds = logits.sigmoid().to('cpu').numpy()
            labs = label_ids.to('cpu').numpy()
            if initRes == True:
                res = preds
                allLabs = labs
                initRes = False
            else:
                res = numpy.concatenate((res, preds))
                allLabs = numpy.concatenate((allLabs, labs))
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        self.predictions = res
        de = datetime.datetime.now()
        print("Test dataset containing %d documents predicted in %s\n" % (len(eval_examples), showTime(ds, de)))
        if self.Config["runfor"] != "crossvalidation":
            self.saveResources("torch")
        self.getMetrics()

    def saveResources(self, type):
        self.resources["id"] = str(self.Config["modelid"])
        self.resources["modelPath"] = fullPath(self.Config, "bertoutpath", opt="name")
        self.resources["modelType"] = type
        if not "ptBertModel" in self.Config["resources"]:
            self.Config["resources"]["ptBertModel"] = self.arsg.bert_model
            self.Config["resources"]["vocabPath"] = self.vocabPath
        self.resources["ptBertModel"] = "yes"
        self.resources["handleType"] = "bert"

    def launchCrossValidation(self):
        print ("Start cross-validation...")
        ds = datetime.datetime.now()
        self.cvDocs = self.Config["traindocs"] + self.Config["testdocs"]
        random.shuffle(self.cvDocs)
        self.keyTrain = "cvtraindocs"
        self.keyTest = "cvtestdocs"
        pSize = len(self.cvDocs) // self.kfold
        ind = 0
        f1 = 0
        for i in range(self.kfold):
            print ("Cross-validation, cycle %d from %d..."%((i+1), self.kfold))
            if i == 0:
                self.Config["cvtraindocs"] = self.cvDocs[pSize:]
                self.Config["cvtestdocs"] = self.cvDocs[:pSize]
            elif i == self.kfold - 1:
                self.Config["cvtraindocs"] = self.cvDocs[:ind]
                self.Config["cvtestdocs"] = self.cvDocs[ind:]
            else:
                self.Config["cvtraindocs"] = self.cvDocs[:ind] + self.cvDocs[ind+pSize:]
                self.Config["cvtestdocs"] = self.cvDocs[ind:ind+pSize]
            ind += pSize
            self.prepareData()
            self.model = self.createModel()
            self.trainModel()
            self.testModel()
            cycleF1 = self.metrics["all"]["f1"]
            print ("Resulting F1-Measure: %f\n"%(cycleF1))
            if cycleF1 > f1:
                if self.Config["cvsave"]:
                    self.saveDataSets()
                f1 = cycleF1
        de = datetime.datetime.now()
        print ("Cross-validation is done in %s"%(showTime(ds, de)))
        print ("The best result is %f"%(f1))
        print ("Corresponding data sets are saved in the folder %s"%(fullPath(self.Config, "cvpath")))


