from typing import Optional
from pathlib import Path
import flux
import flux.job
import flux.resource
import flux.kvs
import os
import sys

def get_total_cores(handle):
    return flux.resource.ResourceSet(flux.kvs.get(handle, "resource.R")).ncores

def get_biggest_host(handle) -> tuple[str, int]:
    R = flux.resource.ResourceSet(flux.kvs.get(handle, "resource.R"))
    biggest_host = ""
    biggest_cores = 0
    for r in R.ranks:
        rank = R.copy_ranks(r)
        host_name = str(rank.nodelist[0])
        cores = rank.ncores
        if cores > biggest_cores:
            biggest_host = host_name
            biggest_cores = cores
    return biggest_host, biggest_cores

def ensure_flux():
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


def create_clustcc_head_jobspec(
        *,
        server_cores: int,
        logging_level: str,
        logging_prefix: str,
        port_file: str,
        fifo_path: str,
        hostname: Optional[str] = None,
        stdout_filename: str = "server.out",
        stderr_filename: str = "server.err",
        cwd: Optional[str] = None,
        env: Optional[dict] = None
) -> flux.job.JobspecV1:
    args = [
        "clustcc-head",
        "--logging-level",  logging_level,
        "--logging-prefix", logging_prefix,
        "--port-file", port_file,
        "--local-concurrent-tasks", str(server_cores),
        "--signal-pipe", fifo_path
    ]
    jobspec = flux.job.JobspecV1.from_command(
        args,
        num_tasks=1,
        cores_per_task=server_cores,
        num_nodes=1
    )
    jobspec.setattr("system.constraints", {"hostlist": [hostname]})
    jobspec.cwd = os.getcwd() if cwd is None else cwd
    jobspec.environment = dict(os.environ) if env is None else env
    jobspec.stdout = os.path.join(logging_prefix, stdout_filename)
    jobspec.stderr = os.path.join(logging_prefix, stderr_filename)
    return jobspec

def create_clustcc_worker_jobspec(
        *,
        worker_cores: int,
        logging_level: str,
        logging_prefix: str,
        port_file: str,
        stdout_filename: str = "worker.out",
        stderr_filename: str = "worker.err",
        cwd: Optional[str] = None,
        env: Optional[dict] = None
) -> flux.job.JobspecV1:
    args = [
        "clustcc-worker",
        "--logging-level",  logging_level,
        "--logging-prefix", logging_prefix,
        "--port-file", port_file,
    ]
    jobspec = flux.job.JobspecV1.from_command(
        args,
        num_tasks=worker_cores,
        cores_per_task=1,
    )
    jobspec.cwd = os.getcwd() if cwd is None else cwd
    jobspec.environment = dict(os.environ) if env is None else env
    jobspec.stdout = os.path.join(logging_prefix, stdout_filename)
    jobspec.stderr = os.path.join(logging_prefix, stderr_filename)
    return jobspec
