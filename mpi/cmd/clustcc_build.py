from argparse import ArgumentParser
from pathlib import Path
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
    from spack.extensions.mpi.concretize import concretize_with_clustcc
    from spack.extensions.mpi.jsonl import read_specs_from_jsonl, write_specs_to_jsonl
    from spack.extensions.mpi.flux_utils import get_total_cores, get_biggest_host, ensure_flux, create_clustcc_head_jobspec, create_clustcc_worker_jobspec
except ImportError as e:
    from concretize import concretize_with_clustcc
    from jsonl import read_specs_from_jsonl, write_specs_to_jsonl
    from flux_utils import get_total_cores, get_biggest_host, ensure_flux, create_clustcc_head_jobspec, create_clustcc_worker_jobspec

level = "long"
description = "distributed builds on a cluster"
section = "build packages"

_WORKER_ENV_VAR = "SPACK_CLUSTCC_INSTALL_WORKER"
def _is_worker() -> bool:
    return _WORKER_ENV_VAR in os.environ
            
def _build_installer_command(worker_cores, installer_cores):
    final = []
    for arg in sys.argv:
        if arg == "clustcc-build":
            final.append(arg)
            final.append("-j"+str(worker_cores))
            final.append("-p"+str(installer_cores))
        else:
            final.append(arg)
    return final

def _orchestrate(args):
    assert "FLUX_URI" in os.environ, "Must orchestrate from within a flux instance"
    handle = flux.Flux()
    total_cores = get_total_cores(handle)
    biggest_host, biggest_cores = get_biggest_host(handle)
    server_cores = max(1, biggest_cores * int(args.server_core_percentage) // 100)
    installer_cores = max(1, biggest_cores * int(args.installer_core_percentage) // 100)
    assert server_cores + installer_cores <= biggest_cores
    worker_cores = total_cores - server_cores - installer_cores
    assert worker_cores >= 1, "Not enough cores to spawn workers"
    fifo_path = str(Path(args.port_file).parent / "clustcc_fifo")
    os.mkfifo(fifo_path)
    Path(args.logging_prefix).mkdir(parents=True, exist_ok=True)
    server_spec = create_clustcc_head_jobspec(
        server_cores=server_cores,
        logging_level=args.logging_level,
        logging_prefix=args.logging_prefix,
        port_file=args.port_file,
        fifo_path=fifo_path,
        hostname=biggest_host
    )
    worker_spec = create_clustcc_worker_jobspec(
        worker_cores=worker_cores,
        logging_level=args.logging_level,
        logging_prefix=args.logging_prefix,
        port_file=args.port_file,
    )
    installer_spec = flux.job.JobspecV1.from_command(
        _build_installer_command(worker_cores, installer_cores),
        num_tasks=1,
        cores_per_task=installer_cores
    )
    installer_spec.cwd = os.getcwd()
    installer_spec.environment = {**os.environ, _WORKER_ENV_VAR: fifo_path}
    installer_spec.stdout = os.path.join(args.logging_prefix, "installer.out")
    installer_spec.stderr = os.path.join(args.logging_prefix, "installer.err")
    installer_spec.setattr("system.constraints", {"hostlist": [biggest_host]})
    server_jid = flux.job.submit(handle, server_spec)
    installer_jid = flux.job.submit(handle, installer_spec)
    worker_jid = flux.job.submit(handle, worker_spec)
    flux.job.result(handle, installer_jid)
    flux.job.cancel(handle, server_jid)
    flux.job.cancel(handle, worker_jid)

    
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
    arguments.add_common_arguments(parser, ["specs", "jobs", "concurrent_packages"])
    parser.add_argument("--spec-jsonl", type=Path, help="Concretized spec to test")
    parser.add_argument(
        "--logging-level",
        choices=["debug", "warning", "error", "none"],
        default="none",
        help="Level for debugging",
    )
    parser.add_argument(
        "--logging-prefix",
        type=str,
        default="logs/",
        help="Directory to store logs",
    )
    parser.add_argument(
        "--port-file",
        type=str,
        help="File where port name will be published to link processes",
    )
    parser.add_argument(
        "--server-core-percentage",
        default="25",
        help="Percentage of the largest allocated node used for the server"
    )
    parser.add_argument(
        "--installer-core-percentage",
        default="10",
        help="Percentage of the largest allocated node used for the installer"
    )


def clustcc_build(parser, args):
    ensure_flux()
    if _is_worker():
        fifo_path = os.environ.get(_WORKER_ENV_VAR)
        assert fifo_path is not None
        packages = _concretize_or_read_jsonl(args.spec_jsonl, args.specs)
        installer = _get_installer_from_config()
        try:
            with open(fifo_path, "r") as fp:
                fp.read()
            os.unlink(fifo_path)
            installer(packages).install()
        except Exception as e:
                raise e
        finally:
            if os.path.exists(args.port_file):
                os.remove(args.port_file)
                
    else:
        _orchestrate(args)
