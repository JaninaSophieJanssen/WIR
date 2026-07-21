import logging
import sys

logger = logging.getLogger("topic_gen")
logger.setLevel("WARNING")
log_handler = logging.StreamHandler(sys.stderr)
log_handler.setFormatter(
    logging.Formatter(
        "[%(name)s] [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s"
    )
)
logger.addHandler(log_handler)
