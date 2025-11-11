import logging
from multiprocessing.queues import Queue
from subprocess import CompletedProcess, run, PIPE
from concurrent.futures import ProcessPoolExecutor
# TODO: Make modular import
try:
    from spack.extensions.mpi.task import (
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerTask,
        parse_task_from_message,
        refine_compiler_task,
    )
    from spack.extensions.mpi.constants import HEAD_NODE_LOGGER_NAME
except:
    from task import (
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerTask,
        parse_task_from_message,
        refine_compiler_task,
    )
    from constants import HEAD_NODE_LOGGER_NAME

logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)

def run_local_compiler_task(task: LocalCompilerTask):
    res = run(
        task.cmd,
        stdout=PIPE,
        stderr=PIPE,
        cwd=task.working_dir,
    )
    with open(task.output_fifo, "ab") as f:
        f.write(res.returncode.to_bytes())
        if len(res.stdout) > 0:
            f.write(res.stdout)
        elif len(res.stderr) > 0:
            f.write(res.stderr)


def run_local_preprocessor_task(
    task: LocalPreprocessorTask, local_task_queue: Queue, remote_task_queue: Queue
):
    """
    Mocking where everything is local
    """
    local_task_queue.put(
        LocalCompilerTask(task.working_dir, task.output_fifo, task.orig_cmd)
    )
    return


def task_server(
    local_task_queue: Queue, remote_task_queue: Queue, concurrent_tasks: int
):
    try:
        with ProcessPoolExecutor(int(concurrent_tasks)) as executor:
            while True:
                task = local_task_queue.get()
                if isinstance(task, LocalCompilerTask):
                    logger.debug(f"Running local compiler task: {task}")
                    executor.submit(run_local_compiler_task, task)
                elif isinstance(task, LocalPreprocessorTask):
                    logger.debug(f"Running local preprocessing task: {task}")
                    executor.submit(
                        run_local_preprocessor_task,
                        task,
                        local_task_queue,
                        remote_task_queue,
                    )
                else:
                    raise Exception(f"Unrecognized task: {task}")
    except EOFError:
        return
