from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from spack.cmd.common import arguments
import tempfile

import os
import flux
import flux.job
import json
import spack.config
import time
import sys

from spack.spec import Spec
import spack.store
try:
    from spack.extensions.mpi.flux_utils import get_total_cores, get_biggest_host, ensure_flux, create_clustcc_head_jobspec, create_clustcc_worker_jobspec
    from spack.extensions.mpi.concretize import _ensure_clustcc_gcc
except:
    from flux_utils import get_total_cores, get_biggest_host, ensure_flux, create_clustcc_head_jobspec, create_clustcc_worker_jobspec
    from concretize import _ensure_clustcc_gcc


level = "long"
description = "benchmark clustcc builds"
section = "build packages"

SETUP_ENV_VAR = "CLUSTCC_SETUP_FIFO"
INSTALL_ENV_VAR = "CLUSTCC_INSTALL_INDEX"

def _build_installer_jobspec(
        *,
        input_index: int,
        run_index: int,
        logging_prefix: str,
        hostname: str,
        installer_cores: int,
        worker_cores: int,
) -> flux.job.JobspecV1:
    args = []
    for arg in sys.argv:
        if arg == "clustcc-benchmark":
            args.append(arg)
            args.append("-j"+str(worker_cores))
            args.append("-p"+str(installer_cores))
        else:
            args.append(arg)
    jobspec = flux.job.JobspecV1.from_command(
        args,
        num_tasks=1,
        cores_per_task=installer_cores
    )
    jobspec.setattr("system.constraints", {"hostlist": [hostname]})
    jobspec.cwd = os.getcwd()
    jobspec.environment = {**os.environ, INSTALL_ENV_VAR: str(input_index)}
    jobspec.stdout = os.path.join(logging_prefix, f"installer_{input_index}_{run_index}.out")
    jobspec.stderr = os.path.join(logging_prefix, f"installer_{input_index}_{run_index}.err")
    return jobspec
    


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
    with open(args.spec_jsonl, "r") as f:
        numspecs = 0
        for _ in f:
            numspecs += 1
    
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
    installer_setup_spec = _build_installer_jobspec(
        input_index=-1,
        run_index=-1,
        logging_prefix=args.logging_prefix,
        hostname=biggest_host,
        installer_cores=installer_cores,
        worker_cores=worker_cores,
    )
    installer_setup_spec.environment = {**os.environ, SETUP_ENV_VAR: fifo_path}
    server_jid = flux.job.submit(handle, server_spec)
    installer_setup_jid = flux.job.submit(handle, installer_setup_spec)
    worker_jid = flux.job.submit(handle, worker_spec)
    flux.job.result(handle, installer_setup_jid)
    for ind in range(numspecs):
        for run in range(int(args.runs_per_spec)):
            installer_spec = _build_installer_jobspec(
                input_index=ind,
                run_index=run,
                logging_prefix=args.logging_prefix,
                hostname=biggest_host,
                installer_cores=installer_cores,
                worker_cores=worker_cores,
            )
            flux.job.result(handle, flux.job.submit(handle, installer_spec))
    flux.job.cancel(handle, server_jid)
    try:
        flux.job.cancel(handle, worker_jid)
    except:
        print("Graceful shutdown complete")
            
def _is_install() -> bool:
    return INSTALL_ENV_VAR in os.environ

def _is_setup() -> bool:
    return SETUP_ENV_VAR in os.environ

def setup_parser(parser: ArgumentParser):
    arguments.add_common_arguments(parser, ["jobs", "concurrent_packages"])
    parser.add_argument("--spec-jsonl", type=str, help="Concretized specs to test")
    parser.add_argument("--output-json", type=str, help="Output file for benchmarked specs")
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
    parser.add_argument(
        "--runs-per-spec",
        default="1",
        help="Number of timing runs per spec"
    )
    
def _get_installer_from_config():
    if spack.config.get("config:installer", "old") == "new":
        from spack.new_installer import PackageInstaller
    else:
        from spack.installer import PackageInstaller
    return PackageInstaller

def clustcc_benchmark(parser, args):
    ensure_flux()
    if _is_setup():
        # Here we just wait for the clustcc to get setup and return
        # this ensures we only do this once 
        clustcc_fifo = os.environ.get(SETUP_ENV_VAR)
        assert clustcc_fifo is not None
        with open(clustcc_fifo, "r") as f:
            f.read()
        
    elif _is_install():
        input_index = os.environ.get(INSTALL_ENV_VAR)
        assert input_index is not None
        input_index = int(input_index)
        input_json_data = None
        with open(args.spec_jsonl, "r") as f:
            for (i, line) in enumerate(f):
                if i == input_index:
                    input_json_data = json.loads(line)
                    break
        assert input_json_data is not None, f"Failed to load json data for index {input_index}"
        abstract_spec = Spec.from_dict(input_json_data['abstract'])
        concrete_spec = Spec.from_dict(input_json_data['concrete'])
        if Path(args.output_json).exists():
            with open(args.output_json, "r") as f:
                output_data = json.load(f)
        else:
            output_data = {}

        installer = _get_installer_from_config()
        with tempfile.TemporaryDirectory() as td:
            with spack.store.use_store(td):
                _ensure_clustcc_gcc(concrete_spec['clustcc-gcc'])
                start = time.time()
                installer([concrete_spec.package]).install()
                end = time.time()
                elapsed = end - start;
                hash = concrete_spec.dag_hash()
                # default dict would be cleaner, but tricky with json
                if hash in output_data:
                    output_data[hash].append(elapsed)
                else:
                    output_data[hash] = [elapsed]
        with open(args.output_json, "w") as f:
            json.dump(output_data, f)
    else:
        _orchestrate(args)
