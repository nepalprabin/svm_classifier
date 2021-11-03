import logging
from time import gmtime, strftime
# >>> strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

def setup_logger(name) -> logging.Logger:
    FORMAT = "[%(name)s %(module)s:%(lineno)s]\n\t %(message)s \n"
    TIME_FORMAT = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    logging.basicConfig(
        format=FORMAT, datefmt=TIME_FORMAT, level=logging.INFO, filename="APILOG.log"
    )

    logger = logging.getLogger(name)
    return logger


# in any file that import fn setup_logger from the above 'logger_config.py', you can set up local logger like:
# local_logger = setup_logger(__name__)

# local_logger.info(
#     f"I am writing to file {FILENAME}. If that file did not exist, it would be automatically created. Here, you can change me to write about the call to a specific route function etc."
# )