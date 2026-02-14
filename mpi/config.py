import json
from typing import TypedDict, Optional
from pathlib import Path


class ClustccConfig(TypedDict):
    slurm_partition: str
    slurm_pmi: str
    slurm_time: str
    slurm_batch_file: str
    launch_file: str
    total_cpus: int
    flux_exe: Path
    spack_exe: Path
    spack_env: Path
    port_file: Path
    specs: str
    num_workers: int
    head_rank_cpus: int
    head_rank_tasks: int
    worker_rank_cpus: int
    logging_level: str
    logging_prefix: Path


# FLUX_LAUNCH_TEMPLATE="""./flux_launch.py \\
#     --spack-exe {spack_exe} \\
#     --spack-env {spack_env} \\
#     --port-file {port_file} \\
#     --specs {specs} \\
#     --num-workers {num_workers} \\
#     --head-rank-cpus {head_rank_cpus} \\
#     --head-rank-tasks {head_rank_tasks} \\
#     --worker-rank-cpus {worker_rank_cpus} \\
#     --logging-level {logging_level} \\
#     --logging-prefix {logging_prefix} && flux queue drain"""
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition={slurm_partition}
#SBATCH --ntasks={total_cpus}
#SBATCH --cpus-per-task=1
#SBATCH --time={slurm_time}
srun --mpi={slurm_pmi} \\
     -n\"$SLURM_NUM_NODES\"\\
     -N\"$SLURM_NUM_NODES\"\\
     {flux_exe} start ./{launch_file}"""

SHELL_LAUNCH_TEMPLATE = """#!/bin/bash
CLUSTCC=\"{spack_exe} \\
    -e {spack_env} \\
    clustcc \\
    --logging-level {logging_level} \\
    --logging-prefix {logging_prefix} \\
    --port-file {port_file}\"
SPECS={specs}
HEAD_CMD=\"$CLUSTCC head -j{num_workers} --local-concurrent-tasks={head_rank_tasks}\"
WORKER_CMD=\"$CLUSTCC worker\"
rm -f {port_file}
flux run -n1 --cores-per-task={head_rank_cpus} $HEAD_CMD \"$SPECS\" &
flux run -n{num_workers} --cores-per-task={worker_rank_cpus} $WORKER_CMD &
wait
"""


def _map_or_raise(data, key, fn=None):
    if key in data:
        if fn is not None:
            return fn(data[key])
        else:
            return data[key]
    else:
        raise Exception(f"Required key: {key} not present in config")


def gen_empty_config_file(file: Optional[Path]):
    empty_config = {
        "slurm_partition": "partition",
        "slurm_pmi": "pmi2",
        "slurm_time": "00:00:00",
        "slurm_batch_file": "clustcc.sbatch",
        "launch_file": "launch.sh",
        "total_cpus": 2,
        "flux_exe": "/path/to/flux",
        "spack_exe": "/path/to/spack/bin/spack",
        "spack_env": "/path/to/env/providing/necessary/deps/for/clustcc",
        "port_file": "/path/to/file/where/head/rank/will/publish/intercomm/port",
        "specs": "zlib hdf5 mpich",
        "num_workers": 1,
        "head_rank_cpus": 1,
        "head_rank_tasks": 1,
        "worker_rank_cpus": 1,
        "logging_level": "debug",
        "logging_prefix": "/path/to/where/logs/should/be/stored",
    }
    if file:
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, "w") as f:
            json.dump(empty_config, f, indent=2)
    else:
        print(json.dumps(empty_config, indent=2))


def parse_config_file(file: Path) -> ClustccConfig:
    """
    Mom: "We have Pydantic at home"
    Pydantic at home:
    """
    with open(file, "r") as f:
        raw_data = json.load(f)
    return {
        "slurm_partition": _map_or_raise(raw_data, "slurm_partition"),
        "slurm_pmi": _map_or_raise(raw_data, "slurm_pmi"),
        "slurm_time": _map_or_raise(raw_data, "slurm_time"),
        "slurm_batch_file": _map_or_raise(raw_data, "slurm_batch_file"),
        "launch_file": _map_or_raise(raw_data, "launch_file"),
        "total_cpus": _map_or_raise(raw_data, "total_cpus", int),
        "flux_exe": _map_or_raise(raw_data, "flux_exe", Path),
        "spack_exe": _map_or_raise(raw_data, "spack_exe", Path),
        "spack_env": _map_or_raise(raw_data, "spack_env", Path),
        "port_file": _map_or_raise(raw_data, "port_file", Path),
        "specs": _map_or_raise(raw_data, "specs"),
        "num_workers": _map_or_raise(raw_data, "num_workers", int),
        "head_rank_cpus": _map_or_raise(raw_data, "head_rank_cpus", int),
        "head_rank_tasks": _map_or_raise(raw_data, "head_rank_tasks", int),
        "worker_rank_cpus": _map_or_raise(raw_data, "worker_rank_cpus", int),
        "logging_level": _map_or_raise(raw_data, "logging_level"),
        "logging_prefix": _map_or_raise(raw_data, "logging_prefix", Path),
    }


def gen_launch_files(config: ClustccConfig, output_dir: Path, is_slurm: bool = True):
    launch_text = SHELL_LAUNCH_TEMPLATE.format(**config)
    with open(output_dir / config["launch_file"], "w") as f:
        f.write(launch_text)
    if is_slurm:
        batch_text = SBATCH_TEMPLATE.format(**config)
        with open(output_dir / config["slurm_batch_file"], "w") as f:
            f.write(batch_text)
