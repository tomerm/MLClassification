import datetime
import logging
import General.settings as settings
from Utils.utils import updateParams

from Models.collector import job_collector
from Preprocess.preprocess import job_preprocessor
from WordEmbedding.embedding import job_word_embedding
from Data.data import job_data_loader
from Models.controller import job_model_controller

logger = logging.getLogger(__name__)

jobs_def = {
    "P": (job_preprocessor,"preprocess"),
    "W": (job_word_embedding,"word_embedding"),
    "D": (job_data_loader,"data"),
    "M": (job_model_controller,"model"),
    "C": (job_collector,"")
}

def run():
    for job in settings.jobs_list:
        logger.info(datetime.datetime.now())
        logger.info(" Start task " + job[0])
        func = jobs_def[job[0]][0]
        kwargs = job[1]
        job_config_name = jobs_def[job[0]][1]
        if job_config_name:
            job_config = settings.def_config[job_config_name]
        else:
            job_config = {}
        updateParams(settings.Config, job_config, kwargs)			
        func()
