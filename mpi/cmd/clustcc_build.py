from argparse import ArgumentParser
from pathlib import Path
from subprocess import Popen
from tempfile import mkdtemp
from typing import Optional

import flux
import flux.job
import flux.resource
import flux.kvs
import os
import sys


import spack.config
from spack.cmd import parse_specs
from spack.cmd.common import arguments
from spack.package_base import PackageBase
from spack.spec import EMPTY_SPEC
try:
    from spack.extensions.mpi.config import (
        parse_config_file,
        gen_launch_files,
        gen_empty_config_file,
    )
    from spack.extensions.mpi.concretize import concretize_with_clustcc
    from spack.extensions.mpi.jsonl import read_specs_from_jsonl, write_specs_to_jsonl
except ImportError as e:
    from config import parse_config_file, gen_launch_files, gen_empty_config_file
    from concretize import concretize_with_clustcc
    from jsonl import read_specs_from_jsonl, write_specs_to_jsonl

level = "long"
description = "distributed builds on a cluster"
section = "build packages"

_WORKER_ENV_VAR = "SPACK_CLUSTCC_INSTALL_WORKER"
def _is_worker() -> bool:
    return _WORKER_ENV_VAR in os.environ
def _ensure_flux():
    if "FLUX_URI" not in os.environ:
        nnodes = os.environ.get("SLURM_NNODES")
        if nnodes is None:
            msg = (
                "SLURM_NNODES and FLUX_URI environment variables are not present, "
                "spack clustcc-build must be run under either a slurm or flux allocation"
            )
            raise Exception(msg)
        cmd = [
            "srun", f"-N{nnodes}", f"-n{nnodes}", "flux", "start", *sys.argv
        ]
        os.execvp("srun", cmd)

def _get_total_cores(handle) -> int:
    return flux.resource.ResourceSet(flux.kvs.get(handle, "resource.R")).ncores

def _get_max_node_cores(handle) -> int:
    R = flux.resource.ResourceSet(flux.kvs.get(handle, "resource.R"))
    return max(R.copy_ranks(r).ncores for r in R.ranks)

def _orchestrate(args):
    assert "FLUX_URI" in os.environ, "Must orchestrate from within a flux instance"
    handle = flux.Flux()
    flux_log_dir = Path(args.logging_prefix) / "flux"
    flux_log_dir.mkdir(parents=True, exist_ok=True)
    total_cores = _get_total_cores(handle)
    max_single_node_cores = _get_max_node_cores(handle)
    server_cores = min(max(1, int(args.server_core_percentage) * total_cores // 100), max_single_node_cores)
    installer_cores = 1
    worker_cores = total_cores - server_cores - installer_cores
    assert worker_cores >= 1, "Not enough cores to spawn workers"
    fifo_path = os.path.join(str(Path(args.port_file).parent), "clustcc_fifo")
    os.mkfifo(fifo_path)
    common_args = [
        "--logging-level", args.logging_level,
        "--logging-prefix", args.logging_prefix,
        "--port-file", args.port_file,
    ]
    server_spec = flux.job.JobspecV1.from_command(
        ["clustcc-head"] + common_args +
        [
            "--local-concurrent-tasks", args.local_concurrent_tasks,
            "--signal-pipe", fifo_path
        ],
        num_tasks=1,
        cores_per_task=server_cores,
        num_nodes=1,
    )
    server_spec.cwd = os.getcwd()
    server_spec.environment = dict(os.environ)
    server_spec.setattr_shell_option("cpu-affinity", "off")
    server_spec.stdout = str(flux_log_dir / "server.out")
    server_spec.stderr = str(flux_log_dir / "server.err")
    worker_spec = flux.job.JobspecV1.from_command(
        ["clustcc-worker"] + common_args,
        num_tasks=worker_cores,
        cores_per_task=1
    )
    worker_spec.cwd = os.getcwd()
    worker_spec.environment = dict(os.environ)
    worker_spec.stdout = str(flux_log_dir / "worker.out")
    worker_spec.stderr = str(flux_log_dir / "worker.err")
    server_jid = flux.job.submit(handle, server_spec)
    worker_jid = flux.job.submit(handle, worker_spec)
    installer_spec = flux.job.JobspecV1.from_command(
        sys.argv,
        num_tasks=1,
        cores_per_task=installer_cores
    )
    installer_spec.cwd = os.getcwd()
    installer_spec.environment = {**os.environ, _WORKER_ENV_VAR: "1"}
    installer_spec.stdout = str(flux_log_dir / "installer.out")
    installer_spec.stderr = str(flux_log_dir / "installer.err")
    with open(fifo_path, "r") as fp:
        server_hostname = fp.read()
    installer_spec.setattr("system.constraints", {"hostlist": [server_hostname]})
    installer_jid = flux.job.submit(handle, installer_spec)
    flux.job.result(handle, installer_jid)
    flux.job.cancel(handle, server_jid)
    try:
        flux.job.cancel(handle, worker_jid)
    except:
        flux.job.result(handle, worker_jid)

    
def _concretize_or_read_jsonl(spec_jsonl: Optional[Path], spec_strs: list[str]) -> list[PackageBase]:
    if spec_jsonl is not None and spec_jsonl.exists():
        packages = []
        spec_pairs = read_specs_from_jsonl(spec_jsonl)
        for (_, concrete) in spec_pairs:
            print(concrete, flush=True)
            if concrete != EMPTY_SPEC:
                packages.append(concrete.package)
    else:
        assert len(spec_strs) > 0, "Must build at least one spec"
        abstract_specs = parse_specs(spec_strs)
        concretized_specs = concretize_with_clustcc(abstract_specs)
        if spec_jsonl is not None:
            write_specs_to_jsonl(abstract_specs, concretized_specs, spec_jsonl)
        packages = [c.package for c in concretized_specs]
    return packages

def _get_installer_from_config():
    if spack.config.get("config:installer", "old") == "new":
        from spack.new_installer import PackageInstaller
    else:
        from spack.installer import PackageInstaller
    return PackageInstaller
        
    
def setup_parser(parser: ArgumentParser):
    """
    Required method for configuring parser for spack command
    """
    subparsers = parser.add_subparsers(dest="subcommand")
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument(
        "config_mode",
        choices=["new", "scripts"],
        help="Generate new config or generate scripts from existing config",
    )
    config_parser.add_argument(
        "config_file",
        help="Path to config json file, created if it does not exist",
        type=Path,
    )
    config_parser.add_argument(
        "--is-slurm",
        action="store_true",
        help="Generate a slurm batch file rather than just a call to launch file",
    )
    install_parser = subparsers.add_parser("install")
    arguments.add_common_arguments(install_parser, ["specs", "jobs"])
    install_parser.add_argument("--spec-jsonl", type=Path, help="Concretized spec to test")
    install_parser.add_argument(
        "--local-concurrent-tasks", default="1", help="Concurrent tasks running on head node"
    )
    install_parser.add_argument(
        "--logging-level",
        choices=["debug", "warning", "error", "none"],
        default="none",
        help="Level for debugging",
    )
    install_parser.add_argument(
        "--logging-prefix",
        type=str,
        default="logs/",
        help="Directory to store logs",
    )
    install_parser.add_argument(
        "--port-file",
        type=str,
        required=True,
        help="File where port name will be published to link processes",
    )
    install_parser.add_argument(
        "--server-core-percentage",
        default="10",
    )


def clustcc_build(parser, args):
    if args.subcommand == "install":
        _ensure_flux()
        if _is_worker():
            packages = _concretize_or_read_jsonl(args.spec_jsonl, args.specs)
            installer = _get_installer_from_config()
            try:
                installer(packages).install()
            except Exception as e:
                raise e
        else:
            _orchestrate(args)
    elif args.subcommand == "config":
        if args.config_mode == "new":
            gen_empty_config_file(args.config_file)
            if args.config_file:
                print(
                    f"An empty config file has been generated at: {args.config_file.absolute()}"
                )
        else:
            gen_launch_files(
                parse_config_file(args.config_file),
                Path(os.getcwd()),
                is_slurm=args.is_slurm,
            )
