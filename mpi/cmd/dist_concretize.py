from mpi4py import MPI  # noqa: E402
from collections import defaultdict
import io
import os
from spack.spec import Spec
import spack.repo
import spack.config
import spack.concretize
import tarfile
import spack.store
import spack.solver.reuse
from spack.database import _DB_DIRNAME, INDEX_JSON_FILE, _INDEX_VERIFIER_FILE
from pathlib import Path
REPOS_ARCNAME = "spack_repo"
STORE_ARCNAME = "store"
TARGET_ROOT = "temp_config"
CACHE_NAME = "cache"

level = "very long"
section = "concretizer"
description = "TODO"

def pack_spec(s: Spec) -> bytearray:
    json_str = s.to_json()
    assert json_str is not None
    json_bytes = bytearray(json_str.encode())
    return json_bytes


def unpack_spec(spec_json_bytes: bytearray) -> Spec:
    spec_json_str = spec_json_bytes.decode()
    return Spec.from_json(spec_json_str)


def setup_parser(parser):
    return


def tar_repos_and_store(store_arcname, repos_arcname) -> bytes:
    DB_ROOT = os.path.join(spack.store.STORE.root, _DB_DIRNAME)
    files_to_tar = []
    files_to_tar.append(
        (
            os.path.join(DB_ROOT, INDEX_JSON_FILE),
            os.path.join(store_arcname, _DB_DIRNAME, INDEX_JSON_FILE),
        )
    )
    files_to_tar.append(
        (
            os.path.join(DB_ROOT, _INDEX_VERIFIER_FILE),
            os.path.join(store_arcname, _DB_DIRNAME, _INDEX_VERIFIER_FILE)
        )
    )
    for r in spack.repo.PATH.repos:
        arcname = os.path.join(repos_arcname, r.namespace)
        files_to_tar.append((r.root, arcname))
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
        for (file, arcname) in files_to_tar:
            tf.add(file, arcname=arcname)
            
    return tar_bytes.getvalue()

def head_rank():
    tar_bytes = tar_repos_and_store(STORE_ARCNAME, REPOS_ARCNAME)
    MPI.COMM_WORLD.bcast(len(tar_bytes), root=0)
    MPI.COMM_WORLD.Bcast([tar_bytes, MPI.BYTE], root=0)
    spec_buffer_lengths = MPI.COMM_WORLD.gather(None, root=0)
    print(spec_buffer_lengths)
    assert spec_buffer_lengths is not None
    for i in range(MPI.COMM_WORLD.Get_size()-1):
        spec_buffer = bytearray(spec_buffer_lengths[i+1])
        MPI.COMM_WORLD.Recv([spec_buffer, MPI.BYTE], source=i+1)
        zlib_concr = unpack_spec(spec_buffer)
        print(f"{zlib_concr} from rank {i+1}")

        
def worker_rank():
    len_tar_bytes = MPI.COMM_WORLD.bcast(None, root=0)
    tar_bytes = bytearray(len_tar_bytes)
    MPI.COMM_WORLD.Bcast([tar_bytes, MPI.BYTE])
    tar_path = os.path.join(TARGET_ROOT, f"spack-mpi-try-concr{MPI.COMM_WORLD.Get_rank()}")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
        os.makedirs(tar_path, exist_ok=True)
        tf.extractall(tar_path)
    config = {}
    config["config"] = {
        "install_tree" : {"root:" : os.path.join(tar_path,STORE_ARCNAME)},
        "misc_cache:" : os.path.join(tar_path, CACHE_NAME)
    }
    config["repos"] = {}
    for p in (Path(tar_path) / REPOS_ARCNAME).iterdir():
        repo_name = p.name
        repo_path = p.absolute()
        config["repos"][repo_name] = str(repo_path)
    config_scope = spack.config.InternalConfigScope("mpi-worker-scope", config)
    with spack.config.override(config_scope):
        spack.solver.reuse._specs_from_mirror = lambda: []
        zlib_concr = spack.concretize.concretize_one("zlib")
    zlib_spec_buffer = pack_spec(zlib_concr)
    MPI.COMM_WORLD.gather(len(zlib_spec_buffer), root=0)
    MPI.COMM_WORLD.Send([zlib_spec_buffer, MPI.BYTE], dest=0)
    
def dist_concretize(parser, args):
    if MPI.COMM_WORLD.rank == 0:
        head_rank()
    else:
        worker_rank()
