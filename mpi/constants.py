from enum import IntEnum
from pathlib import Path

HEAD_RANK_ID = 0
MQ_NAME = "/spackclustccmq"
MSG_SEP = ";"
MQ_DONE = "Done"
HEAD_NODE_LOGGER_NAME = "head_node_logger"
DEFAULT_PORTFILE = Path("/tmp/spackclustccportfile.txt")


class WorkerResponseTag(IntEnum):
    RESPONSE = 1
    OBJECT_BYTES = 2
