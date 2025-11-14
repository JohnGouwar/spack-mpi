from enum import IntEnum

HEAD_RANK_ID = 0
MQ_NAME = "/spackclustccmq"
MSG_SEP = ";"
MQ_DONE = "Done"
HEAD_NODE_LOGGER_NAME = "head_node_logger"


class WorkerResponseTag(IntEnum):
    RESPONSE = 1
    OBJECT_BYTES = 2
