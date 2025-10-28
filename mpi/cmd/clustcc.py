from multiprocessing import Process
from multiprocessing.connection import Connection, Pipe
from argparse import ArgumentParser
from subprocess import CompletedProcess
from pathlib import Path
import logging

from spack.cmd.common import arguments
from spack.spec import Spec
from spack.concretize import concretize_together
try:
    from spack.extensions.mpi.servers import (
        subprocess_server,
        listener_server,
        installer,
        CompilerTask
    )
    from spack.extensions.mpi.swap import _swap_in_spec
    from spack.extensions.mpi.head_rank import head_rank, setup_head_rank
    from spack.extensions.mpi.worker_rank import worker_rank, setup_worker_rank
    from spack.extensions.mpi.logs import setup_logging
except:
    from servers import subprocess_server, listener_server, installer, CompilerTask 
    from swap import _swap_in_spec
    from logs import setup_logging
    from head_rank import head_rank, setup_head_rank
    from worker_rank import worker_rank, setup_worker_rank

    

def setup_parser(parser: ArgumentParser):
    '''
    Required method for configuring parser for spack command
    '''
    subparsers = parser.add_subparsers(dest="subcommand")
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs"])
    head_parser.add_argument(
        "--mq-name",
        help="Name of the Posix message queue the wrapper will contact",
        default="/spackclustccmq"
    )
    worker_parser = subparsers.add_parser("worker")
    local_test_parser = subparsers.add_parser(
        "test",
        description="Run the various servers locally, mostly for debugging"
    )
    local_test_parser.add_argument("--test-spec-json", help="Concretized spec to test")
    local_test_parser.add_argument("--clustcc-spec-json", help="Concretized wrapper spec")
    local_test_parser.add_argument("--no-install", help="Just setup a server, but don't run the installer", action="store_true")

def setup_local_head_rank(args) -> tuple[Connection, Connection]:
    # Listener proc
    task_queue_read, task_queue_write = Pipe(duplex=False)
    Process(
        target=listener_server,
        args=("/spackclustccmq", task_queue_write,)
    ).start()
    task_queue_write.close()
    # Subprocess server
    parent_subprocess_pipe, child_subprocess_pipe = Pipe()
    Process(
        target=subprocess_server,
        args=(child_subprocess_pipe,)
    ).start()
    # Installer proc
    # Pre-load concretized specs for testing iteration speed
    if not args.no_install:
        with open(args.test_spec_json) as f:
            test_spec = Spec.from_json(f)
        with open(args.clustcc_spec_json) as f:
            clustcc_spec = Spec.from_json(f)
        wrapped_test_spec = _swap_in_spec(
            test_spec,
            {"compiler-wrapper": clustcc_spec},
            {}
        )
        packages = [wrapped_test_spec.package]
        Process(
            target=installer,
            args=("/spackclustccmq", packages)
        ).start()
    return (task_queue_read, parent_subprocess_pipe)
    

def local_head_rank(
        task_queue: Connection,
        subprocess_pipe: Connection
):
    from mpi4py import MPI
    try:
        while True:
            task : CompilerTask = task_queue.recv()
            logging.info(f"Task: {task}")
            subprocess_pipe.send(task.cmd)
            res : CompletedProcess = subprocess_pipe.recv()
            with open(task.output_fifo, "ab") as f:
                f.write(res.returncode.to_bytes())
                if res.returncode != 0:
                    f.write(res.stderr)
                else:
                    f.write(res.stdout)
    except EOFError: # we expect this error from task_queue.recv()
        logging.info(f"We hit EOFerror")
        pass
    except Exception as e:
        logging.warning(f"There was an unexpected exception: {e}")
    finally:
        subprocess_pipe.send(None)
        return
    
    
def clustcc(parser, args):
    with setup_logging(Path("logs.log")):
        if args.subcommand == "head":
            task_queue_read, subprocess_pipe = setup_head_rank(args)
            head_rank(task_queue_read, subprocess_pipe)
        elif args.subcommand == "worker":
            subprocess_pipe = setup_worker_rank(args)
            worker_rank(subprocess_pipe)
        elif args.subcommand == "test":
            # Pre-concretize and cache test specs 
            if not (Path(args.test_spec_json).exists() and
                    Path(args.clustcc_spec_json).exists()):
                test_pair, clustcc_pair = concretize_together([
                    (Spec("zlib"), None),
                    (Spec("clustcc-compiler-wrapper"), None)
                ])
                with open(args.test_spec_json, "w") as f:
                    test_pair[1].to_json(f)
                with open(args.clustcc_spec_json, "w") as f:
                    clustcc_pair[1].to_json(f)
            task_queue_read, subprocess_pipe = setup_local_head_rank(args)
            local_head_rank(task_queue_read, subprocess_pipe)
        
    
