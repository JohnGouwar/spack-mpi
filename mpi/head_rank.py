import logging
from subprocess import CompletedProcess
from typing import Callable, Optional
from mpi.servers import task_server
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from PosixMQ import PosixMQ
from mpi4py import MPI
from time import sleep
from collections import deque
from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe, Queue
from argparse import Namespace
from dataclasses import dataclass
from spack.cmd import parse_specs
from spack.spec import Spec
from pathlib import Path
from queue import Empty
from spack.package_base import PackageBase
from spack.installer import PackageInstaller

try:
    from spack.extensions.mpi.servers import task_server
    from spack.extensions.mpi.swap import concretize_with_clustcc, _swap_in_spec
    from spack.extensions.mpi.constants import (
        HEAD_RANK_ID,
        MQ_NAME,
        MQ_DONE,
        HEAD_NODE_LOGGER_NAME,
    )
    from spack.extensions.mpi.task import (
        parse_task_from_message,
        refine_compiler_task,
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerTask,
        RemoteCompilerResponse,
    )
except:
    from servers import task_server
    from swap import concretize_with_clustcc, _swap_in_spec
    from constants import HEAD_RANK_ID, MQ_NAME, MQ_DONE, HEAD_NODE_LOGGER_NAME
    from task import (
        parse_task_from_message,
        refine_compiler_task,
        LocalCompilerTask,
        LocalPreprocessorTask,
        RemoteCompilerTask,
        RemoteCompilerResponse,
    )

logger = logging.getLogger(HEAD_NODE_LOGGER_NAME)


class HeadRankTaskServer:
    @staticmethod
    def listener_server(mq_name: str, local_task_queue: Queue):
        mq = PosixMQ.create(mq_name)
        try:
            while True:
                task_str = mq.recv()
                logger.debug(f"Listener received: {task_str}")
                if task_str == MQ_DONE:
                    return
                raw_task = parse_task_from_message(task_str)
                logger.debug(f"Listener parsed {task_str} into {raw_task}")
                local_task = refine_compiler_task(raw_task)
                logger.debug(f"Listener refined {raw_task} into {local_task}")
                local_task_queue.put(local_task)
        finally:
            local_task_queue.close()
            mq.unlink()

    @staticmethod
    def installer(packages: list[PackageBase]):
        try:
            PackageInstaller(packages).install()
        except Exception as e:
            logger.debug(f"Installer raised exception: {e}")
            raise e
        finally:
            mq = PosixMQ.open(MQ_NAME)
            mq.send(MQ_DONE, 2)
            mq.close()

    def __init__(
        self,
        specs: list[str],
        spec_json: Optional[Path],
        clustcc_spec_json: Optional[Path],
        concurrent_tasks: int,
    ):
        """
        This must be created before MPI.Init()
        """
        assert not MPI.Is_initialized(), (
            "Cannot initialize MPI befor creating task server"
        )
        self.local_task_queue = Queue()
        self.remote_task_queue = Queue()
        Process(
            target=HeadRankTaskServer.listener_server,
            args=(MQ_NAME, self.local_task_queue),
        ).start()
        Process(
            target=task_server,
            args=(self.local_task_queue, self.remote_task_queue, concurrent_tasks),
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
        Process(target=HeadRankTaskServer.installer, args=(packages,)).start()

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
