import os
import numpy
import random
import torch
import datetime
import logging
from torch.utils.data import TensorDataset, DataLoader as BertDataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from tqdm import tqdm, trange
from Models.base import BaseModel
from Models.bertClassifier import BertForMultiLabelSequenceClassification, \
     convert_examples_to_features, accuracy
from Data.data import compose_tsv
from Models.metrics import print_averaged_metrics
from Utils.utils import get_abs_path, get_formatted_date, test_path
from Models.args import Args
from Models.dataprocessor import DataProcessor
import General.settings as settings

logger = logging.getLogger(__name__)

class BertModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.use_probabilitiesh = True
        self.max_bert_seq_length = 512
        self.device = 'cpu'
        self.n_gpu = torch.cuda.device_count()
        self.model_to_save = None
        if not self.is_correct_path():
            raise Exception
        self.args = Args(get_abs_path(settings.Config, "pretrained_bert_model_path"),
                         get_abs_path(settings.Config, "resulting_bert_files_path")) # model: pytorch_ber.gz
        self.max_seq_length = min(self.max_bert_seq_length, settings.Config["max_seq_len"])
        if settings.Config["type_of_execution"] != "test":
            self.do_train = True
        if settings.Config["type_of_execution"] != "train":
            self.do_eval = True
        self.do_lower_case = False
        self.train_batch_size = min(self.train_batch, 32)
        self.eval_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = self.epochs
        self.warmup_proportion = 0.1
        self.no_cuda = True
        self.local_rank = -1
        self.seed = 42
        self.gradient_accumulation_steps = 1
        self.key_train = "train_docs"
        self.key_test = "test_docs"
        #self.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        if settings.Config["type_of_execution"] != "crossvalidation":
            self.prepare_data()

    def is_correct_path(self):
        test_path(settings.Config, "pretrained_bert_model_path", "Wrong path to archive with pre-trained BERT model. Stop.")
        test_path(settings.Config, "resulting_bert_files_path", "Wrong path to folder with resulting BERT files. Stop.")
        return True

    def prepare_data(self):
        logger.info("Start data preparation...")
        compose_tsv(self, "train")
        compose_tsv(self, "test")
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if os.path.exists(self.args.output_dir) and self.do_train:
            logger.warning("Output directory ({}) already exists and is not empty.".format(self.args.output_dir))
            logger.warning("Its content will be deleted.")
        if self.do_train:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.processor = DataProcessor(settings.dynamic_store["predefined_categories"])
        self.num_labels = len(settings.dynamic_store["predefined_categories"])
        self.label_list = self.processor.get_labels()
        self.vocabPath = os.path.dirname(self.args.bert_model) + "/vocab.txt"
        self.tokenizer = BertTokenizer(self.vocabPath)

    def load_model(self):
        self.model = self.create_model()

    def create_model(self):
        self.train_examples = self.processor.get_train_examples(self.args.data_dir)
        self.num_train_optimization_steps = int(
            len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps)\
                                       * self.num_train_epochs
        model = BertForMultiLabelSequenceClassification.from_pretrained(self.args.bert_model,
                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                            self.local_rank), num_labels=self.num_labels)
        model.to(self.device)
        return model

    def train_model(self):
        logger.info("Start training..")
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
        logger.info("Model is trained in %s" %  (get_formatted_date(ds, de)))
        if settings.Config["type_of_execution"] == "crossvalidation":
            return
        logger.info("Model evaluation...")
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
        all_labs = None
        res = None
        init_res = True
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
            if init_res == True:
                res = preds
                all_labs = labs
                init_res = False
            else:
                res = numpy.concatenate((res, preds))
                all_labs = numpy.concatenate((all_labs, labs))
            tmp_eval_accuracy = accuracy(logits, label_ids, self.rank_threshold)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        eval_accuracy = eval_accuracy / nb_eval_examples
        logger.info("Model accuracy: %.2f"%(eval_accuracy))
        # Save a trained model
        self.model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = get_abs_path(settings.Config, "resulting_bert_files_path", opt="name")
        torch.save(self.model_to_save.state_dict(), output_model_file)
        logger.info("Model is saved in %s"%(output_model_file))

    def test_model(self):
        logger.info("Start testing...")
        logger.info("Rank threshold: %.2f" % (self.rank_threshold))
        ds = datetime.datetime.now()
        if self.model_to_save == None:
            output_model_file = get_abs_path(settings.Config, "resulting_bert_files_path", opt="name")
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
        all_labs = None
        res = None
        init_res = True
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
            if init_res == True:
                res = preds
                all_labs = labs
                init_res = False
            else:
                res = numpy.concatenate((res, preds))
                all_labs = numpy.concatenate((all_labs, labs))
            tmp_eval_accuracy = accuracy(logits, label_ids, self.rank_threshold)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        self.predictions = res
        self.test_labels = all_labs
        de = datetime.datetime.now()
        logger.info("Test dataset containing %d documents predicted in %s\n" % (len(eval_examples), get_formatted_date(ds, de)))
        if settings.Config["type_of_execution"] != "crossvalidation":
            self.prepare_resources_for_runtime("torch")
        self.get_metrics()
        self.save_results()

    def prepare_resources_for_runtime(self, type):
        self.resources["id"] = str(settings.dynamic_store["modelid"])
        self.resources["created_model_path"] = get_abs_path(settings.Config, "resulting_bert_files_path", opt="name")
        self.resources["modelType"] = type
        if not "ptBertModel" in settings.dynamic_store["resources"]:
            settings.dynamic_store["resources"]["ptBertModel"] = self.args.bert_model
            settings.dynamic_store["resources"]["vocabPath"] = self.vocabPath
        self.resources["ptBertModel"] = "True"
        self.resources["handleType"] = "bert"
        self.resources["rank_threshold"] = self.rank_threshold
        settings.dynamic_store["resources"]["models"]["Model" + str(settings.dynamic_store["modelid"])] = self.resources

    def launch_crossvalidation(self):
        logger.info("Start cross-validation...")
        ds = datetime.datetime.now()
        self.cvDocs = settings.dynamic_store["train_docs"] + settings.dynamic_store["test_docs"]
        random.shuffle(self.cvDocs)
        self.key_train = "cross_validations_train_docs"
        self.key_test = "cross_validations_test_docs"
        p_size = len(self.cvDocs) // self.cross_validations_total
        ind = 0
        f1 = 0
        attr_metrics =[]
        for i in range(self.cross_validations_total):
            logger.info("Cross-validation, cycle %d from %d..."%((i+1), self.cross_validations_total))
            if i == 0:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[p_size:]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[:p_size]
            elif i == self.cross_validations_total - 1:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[:ind]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[ind:]
            else:
                settings.dynamic_store["cross_validations_train_docs"] = self.cvDocs[:ind] + self.cvDocs[ind+p_size:]
                settings.dynamic_store["cross_validations_test_docs"] = self.cvDocs[ind:ind+p_size]
            ind += p_size
            self.prepare_data()
            self.model = self.create_model()
            self.train_model()
            self.test_model()
            attr_metrics.append(self.metrics)
            cycle_f1 = self.metrics["all"]["f1"]
            logger.info("Resulting F1-Measure: %f\n"%(cycle_f1))
            if cycle_f1 > f1:
                if settings.Config["save_cross_validations_datasets"]:
                    self.save_data_sets()
                f1 = cycle_f1
        de = datetime.datetime.now()
        logger.info("Cross-validation is done in %s" % get_formatted_date(ds, de))
        print_averaged_metrics(attr_metrics)
        logger.info("The best result is %f"%(f1))
        logger.info("Corresponding data sets are saved in the folder %s"
               % get_abs_path(settings.Config, "cross_validations_datasets_path"))
