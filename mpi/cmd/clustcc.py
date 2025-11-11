import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from argparse import ArgumentParser
from pathlib import Path

from spack.cmd.common import arguments

try:
    from spack.extensions.mpi.head_rank import MpiHeadRank, HeadRankTaskServer
    from spack.extensions.mpi.worker_rank import MpiWorkerRank, ForkServer
except:
    from head_rank import MpiHeadRank, HeadRankTaskServer
    from worker_rank import MpiWorkerRank, ForkServer


def setup_parser(parser: ArgumentParser):
    """
    Required method for configuring parser for spack command
    """
    subparsers = parser.add_subparsers(dest="subcommand")
    head_parser = subparsers.add_parser("head")
    arguments.add_common_arguments(head_parser, ["specs"])
    head_parser.add_argument(
        "--mq-name",
        help="Name of the Posix message queue the wrapper will contact",
        default="/spackclustccmq",
    )
    head_parser.add_argument("--spec-json", help="Concretized spec to test")
    head_parser.add_argument("--clustcc-spec-json", help="Concretized wrapper spec")
    head_parser.add_argument(
        "--local-concurrent-tasks", help="Concurrent tasks running on head node"
    )
    worker_parser = subparsers.add_parser("worker")


def clustcc(parser, args):
    if args.subcommand == "head":
        try:
            hrts = HeadRankTaskServer(
                specs=args.specs,
                spec_json=Path(args.spec_json) if args.spec_json else None,
                clustcc_spec_json=Path(args.clustcc_spec_json)
                if args.clustcc_spec_json
                else None,
                concurrent_tasks=int(args.local_concurrent_tasks),
            )
            MPI.Init()
            MpiHeadRank(hrts).run()
        except Exception as e:
            raise e
        finally:
            if MPI.Is_initialized():
                MPI.Finalize()
    elif args.subcommand == "worker":
        forkserver = ForkServer()
        try:
            MPI.Init()
            MpiWorkerRank(forkserver).run()
        finally:
            MPI.Finalize()
