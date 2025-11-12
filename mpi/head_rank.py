import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from subprocess import PIPE, run
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Optional

from PosixMQ import PosixMQ
from spack.cmd import parse_specs
from spack.installer import PackageInstaller
from spack.package_base import PackageBase
from spack.spec import Spec

import mpi4py # noqa: E402
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402

try:
    from spack.extensions.mpi.compile_commands import parse_compile_command_list
    from spack.extensions.mpi.constants import (
        HEAD_NODE_LOGGER_NAME,
        HEAD_RANK_ID,
        MQ_DONE,
        MQ_NAME,
    )
    from spack.extensions.mpi.logs import LoggingProcess, attach_queue_to_logger
    from spack.extensions.mpi.swap import _swap_in_spec, concretize_with_clustcc
    from spack.extensions.mpi.task import (
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerResponse,
        RemoteCompilerTask,
        parse_task_from_message,
        refine_compiler_task,
    )
except ImportError:
    from compile_commands import parse_compile_command_list
    from constants import HEAD_NODE_LOGGER_NAME, HEAD_RANK_ID, MQ_DONE, MQ_NAME
    from logs import LoggingProcess, attach_queue_to_logger
    from swap import _swap_in_spec, concretize_with_clustcc
    from task import (
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerResponse,
        RemoteCompilerTask,
        parse_task_from_message,
        refine_compiler_task,
    )

# Because of the quirks of ProcessPoolExecutor, sharing queues is very tricky
# The workaround here is to make the queues global variables and then initialize
# them when the task server spawns the initial tasks. This is ugly, but the only
# way to get it to work
global_local_task_queue = None
global_remote_task_queue = None
global_logging_queue = None
def task_server_initializer(local_queue, remote_queue, logging_queue):
    global global_local_task_queue, global_remote_task_queue, global_logging_queue
    global_local_task_queue = local_queue
    global_remote_task_queue = remote_queue
    global_logging_queue = logging_queue
    

def run_local_compiler_task(task: LocalCompilerTask):
    """
    Run tasks that either cannot be distributed (e.g., linking), or compiles that
    failed some portion of remote processing (e.g., failed to parse initial command,
    split preprocessing error, remote command failed with non-zero exit status)
    """
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


def run_local_preprocessor_task(task: LocalPreprocessorTask):
    global global_local_task_queue, global_remote_task_queue, global_logging_queue
    assert global_local_task_queue is not None
    assert global_remote_task_queue is not None
    assert global_logging_queue is not None
    attach_queue_to_logger(global_logging_queue)
    logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
    LOCAL_FALLBACK = LocalCompilerTask(
        task.working_dir, task.output_fifo, task.orig_cmd
    )
    try:
        parsed_command = parse_compile_command_list(task.orig_cmd)
        logger.debug(f"Parsed compile command {task.orig_cmd} as {parsed_command}")
        new_command = list(task.orig_cmd)
        with NamedTemporaryFile("w", delete_on_close=False) as tf:
            tf.close()
            if parsed_command.output_index:
                new_command[parsed_command.output_index] = tf.name
            else:
                new_command.extend(["-o", tf.name])
            new_command = (
                [new_command[0]] + ["-E", "--fdirectives-only"] + new_command[1:]
            )
            logger.debug(f"Running {new_command}")
            res = run(new_command, stdout=PIPE, stderr=PIPE, cwd=task.working_dir)
            if res.returncode == 0:
                with open(tf.name, "r") as f:
                    preproc_text = f.read()
                logger.debug(f"Dispatching remote task for {task.orig_cmd}")
                global_remote_task_queue.put(
                    RemoteCompilerTask(
                        input_file_text=preproc_text,
                        working_dir=task.working_dir,
                        output_fifo=task.output_fifo,
                        orig_cmd=task.orig_cmd,
                    )
                )
            else:
                logger.debug(f"Preprocessing failed for: {task.orig_cmd}, got res: {res}")
                global_local_task_queue.put(LOCAL_FALLBACK)
    except Exception as e:
        logger.debug(f"Exception in preprocessor task: {e}")
        global_local_task_queue.put(LOCAL_FALLBACK)


class HeadRankTaskServer:
    """
    Shutdown order: Installer -> Listener -> Task Server -> MpiHeadRank
    Installer sends MQ_DONE to Listener
    Listener sends None to Task Server Local Queue
    Task Server sends None to MpiRank
    """
    @staticmethod
    def _listener_server(mq_name: str, local_task_queue: Queue):
        mq = PosixMQ.create(mq_name)
        logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
        logger.info("Starting listener daemon")
        try:
            while True:
                task_str = mq.recv()
                logger.debug(f"Listener received: {task_str}")
                if task_str == MQ_DONE:
                    return
                raw_task = parse_task_from_message(task_str)
                local_task = refine_compiler_task(raw_task)
                logger.debug(f"Listener refined: {task_str} into {local_task}")
                local_task_queue.put(local_task)
        finally:
            local_task_queue.put(None)
            local_task_queue.close()
            mq.unlink()

    @staticmethod
    def _installer(packages: list[PackageBase]):
        logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
        logger.info("Starting installer")
        try:
            PackageInstaller(packages).install()
        except Exception as e:
            raise e
        finally:
            mq = PosixMQ.open(MQ_NAME)
            mq.send(MQ_DONE, 2)
            mq.close()

    @staticmethod
    def _task_server_loop(
            local_task_queue: Queue,
            remote_task_queue: Queue,
            concurrent_tasks: int,
            logging_queue: Queue
    ):
        logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)
        logger.info("Starting task server")
        try:
            with ProcessPoolExecutor(
                int(concurrent_tasks),
                initializer=task_server_initializer,
                initargs=(local_task_queue, remote_task_queue, logging_queue),
            ) as executor:
                while True:
                    task = local_task_queue.get()
                    if task is None: # sentinel value for no more work
                        remote_task_queue.put(None)
                        return
                    if isinstance(task, LocalCompilerTask):
                        executor.submit(run_local_compiler_task, task)
                    elif isinstance(task, LocalPreprocessorTask):
                        executor.submit(
                            run_local_preprocessor_task,
                            task,
                        )
                    else:
                        raise Exception(f"Unrecognized task: {task}")
        except EOFError:
            return

    def __init__(
        self,
        specs: list[str],
        spec_json: Optional[Path],
        clustcc_spec_json: Optional[Path],
        concurrent_tasks: int,
        logging_queue: Queue
    ):
        """
        This must be created before MPI.Init()
        """
        assert not MPI.Is_initialized(), (
            "Cannot initialize MPI befor creating task server"
        )
        self.local_task_queue = Queue()
        self.remote_task_queue = Queue()
        self.logging_queue = logging_queue
        LoggingProcess(
            target=HeadRankTaskServer._listener_server,
            args=(MQ_NAME, self.local_task_queue),
            log_queue=self.logging_queue
        ).start()
        LoggingProcess(
            target=HeadRankTaskServer._task_server_loop,
            args=(self.local_task_queue, self.remote_task_queue, concurrent_tasks, self.logging_queue),
            log_queue=self.logging_queue
        ).start()
        # Installer proc
        if (
            spec_json
            and clustcc_spec_json
            and spec_json.exists()
            and clustcc_spec_json.exists()
        ):
            with open(spec_json, "r") as f:
                spec = Spec.from_json(f)
            with open(clustcc_spec_json, "r") as f:
                clustcc_spec = Spec.from_json(f)
            wrapped_test_spec = _swap_in_spec(
                spec, {"compiler-wrapper": clustcc_spec}, {}
            )
            packages = [wrapped_test_spec.package]
        else:
            assert len(specs) > 0, "Must build at least one spec"
            user_specs = parse_specs(specs)
            clustcc_specs = concretize_with_clustcc(user_specs)
            packages = [c.package for c in clustcc_specs]
        LoggingProcess(
            target=HeadRankTaskServer._installer,
            args=(packages,),
            log_queue=self.logging_queue
        ).start()
        

    # The MpiHeadRank either takes tasks to be dispatched remotely or requeues
    # tasks to be tried locally
    def enqueue_local(self, task: LocalCompilerTask):
        self.local_task_queue.put(task)

    def dequeue_remote(self) -> RemoteCompilerTask:
        return self.remote_task_queue.get_nowait()


class MpiHeadRank:
    # [O: rank 1 send, 1: rank 1 recv, 2: rank 2 send, ...]
    # send indices are even, recv indices are hard
    @staticmethod
    def _get_request_indices(rank):
        send_index = 2 * (rank - 1)
        recv_index = send_index + 1
        return (send_index, recv_index)

    @staticmethod
    def _is_send_index(index):
        return index % 2 == 0

    @staticmethod
    def _rank_from_index(index):
        return index // 2

    def __init__(self, task_server: HeadRankTaskServer):
        """
        Requires MPI.Init() has been called
        """
        assert MPI.Is_initialized()
        assert MPI.COMM_WORLD.Get_rank() == HEAD_RANK_ID, (
            f"head rank must be rank {HEAD_RANK_ID}"
        )
        self.world_comm = MPI.COMM_WORLD.Dup()
        self.world_size = self.world_comm.Get_size()
        self.requests = [MPI.REQUEST_NULL for _ in range(2, 2 * self.world_size)]
        self.task_server = task_server

    def _send_task_to_rank(self, dest_rank: int, task: RemoteCompilerTask):
        send_req = self.world_comm.isend(task, dest=dest_rank)
        recv_req = self.world_comm.irecv(source=dest_rank)
        send_index, recv_index = MpiHeadRank._get_request_indices(dest_rank)
        self.requests[send_index] = send_req
        self.requests[recv_index] = recv_req

    def _handle_resp(self, resp: RemoteCompilerResponse):
        if resp.rc == 0:
            assert resp.output_filename is not None, "Must have output filename"
            assert resp.output_bytes is not None, "Must have object bytes"
            with open(resp.output_filename, "wb") as f:
                f.write(resp.output_bytes)
            with open(resp.output_fifo, "wb") as f:
                f.write(resp.rc.to_bytes())
                if resp.stdout:
                    f.write(resp.stdout)
        else:
            # TODO: Log failure and retry locally
            assert resp.cmd is not None, (
                "Failed calls should return command for local retry"
            )

            self.task_server.enqueue_local(
                LocalCompilerTask(resp.working_dir, resp.output_fifo, resp.cmd)
            )
            return

    def run(self):
        idle_workers = deque(range(1, self.world_size))
        # task_queue.recv() will fail with EOFError once its write end has been
        # closed and all data is cleared from it
        try:
            while True:
                # clear idle workers with available tasks
                while len(idle_workers) > 0:
                    try:
                        task = self.task_server.dequeue_remote()
                        worker_rank = idle_workers.popleft()
                        self._send_task_to_rank(worker_rank, task)
                    except Empty:
                        break

                # Process async requests
                inds, resps = MPI.Request.waitsome(self.requests)
                if inds and resps:
                    for i, resp in zip(inds, resps):
                        if not MpiHeadRank._is_send_index(
                            i
                        ):  # do nothing on send indices
                            rank = MpiHeadRank._rank_from_index(i)
                            self._handle_resp(resp)
                            try:
                                task = self.task_server.dequeue_remote()
                                self._send_task_to_rank(rank, task)
                            except Empty:
                                idle_workers.append(rank)

                else:  # requests is all null, sleep and wait for more requests
                    # TODO: Log the sleep
                    sleep(1)

        except EOFError:
            # When we run out of tasks, make sure to complete the ones that are still outstanding
            resps = MPI.Request.waitall(self.requests)
            for resp in resps:
                if resp:
                    self._handle_resp(resp)
            return
