import sys
import General.launcher as launcher
import General.settings as settings
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
    if False == settings.parse_config(sys.argv[1]):
        return
    if len(sys.argv) > 2:
        settings.parse_request(sys.argv[2])
    else:
        settings.parse_request("");
    launcher.run()


if __name__ == "__main__":
    main()

