import sys
import General.launcher as G
import General.settings as settings
from Info.creator import InfoCreator
import logging
import datetime

def main():
    file_name = './' + datetime.datetime.now().strftime("%Y-%m-%d_%I%p") + '.log'
    logging.basicConfig(filename=file_name, filemode="w", format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler())

    if len(sys.argv) == 1:
        logger.error("Missing path to config file. Exit.")
        return

    if False == settings.parse_config_info(sys.argv[1]):
        return
    InfoCreator()

if __name__ == "__main__":
    main()