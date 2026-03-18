from spack.spec import EMPTY_SPEC, Spec
from spack.concretize import concretize_one, concretize_separately
from spack.installer import PackageInstaller
import spack.llnl.util.tty as tty
from spack.bootstrap import ensure_bootstrap_configuration, ensure_clingo_importable_or_raise
import spack.store
import spack.config
import spack.compilers.config
import spack.repo
import spack.util.parallel
from typing import Optional, Union
import importlib
from contextlib import contextmanager
def _ensure_clustcc_gcc(query_spec: Optional[Union[str, Spec]] = None) -> Spec:
    if query_spec is None:
        query_spec = "clustcc-gcc"
    
    clustcc_spec = spack.store.STORE.db.query_one(query_spec, installed=True)
    if clustcc_spec is not None:
        tty.info(f"Already installed {clustcc_spec.format('{name}/{hash:7}')}")
        return clustcc_spec
    else:
        tty.info("Installing clustcc-gcc")
        clustcc_spec = concretize_one(query_spec)
        PackageInstaller([clustcc_spec.package]).install()
        return clustcc_spec



@contextmanager
def require_clustcc(clustcc_spec = None):
    requirements = {"all": {
        "require": "%[when=%c]c=clustcc-gcc %[when=%cxx]cxx=clustcc-gcc"
    }}
    try:
        _ensure_clustcc_gcc(clustcc_spec)
        with spack.config.override("packages", requirements) as c:
            yield c
    finally:
            pass
        
def _best_effort_concr_task(packed_arguments: tuple[int, str]) -> tuple[int, Union[Spec, str]]:
    '''
    Forked concretization task that simply returns None for the spec on failure
    '''
    index, spec_str = packed_arguments
    try:
        with tty.SuppressOutput(
                error_enabled=False,
                msg_enabled=False,
                warn_enabled=False
        ):
            spec = concretize_one(Spec(spec_str), tests=False)
            return index, spec
    except Exception as e:
        return index, str(e)

    
def best_effort_concretize(to_concretize: list[Spec]):
    '''
    This is a best-effort reimplementation of `spack.concretize.concretize_separately`
    where we also return specs that fail to concretize
    '''
    args = [
      (i, str(abstract))
      for i, abstract in enumerate(to_concretize)
    ]
    if len(args) == 0:
        return []
    # Ensure all bootstrapping is done before forking
    try:
        importlib.import_module("clingo")
    except:
        with ensure_bootstrap_configuration():
            ensure_clingo_importable_or_raise()

    # Ensure all global updates are made 
    _ = spack.repo.PATH.provider_index
    _ = spack.compilers.config.all_compilers()
    num_procs = min(len(args), spack.config.determine_number_of_jobs(parallel=True))
    concrete_specs = [EMPTY_SPEC for _ in to_concretize]
    for (i, concrete) in spack.util.parallel.imap_unordered(
            _best_effort_concr_task, args, processes=num_procs, maxtaskperchild=1
    ):
        if isinstance(concrete, Spec):
            tty.info(f"Successfully concretized {to_concretize[i]} to {concrete.format('{name}/{hash:7}')}")
            concrete_specs[i] = concrete
        else:
            tty.info(f"Failed to concretize {to_concretize[i]} with exeception {concrete}")
    return concrete_specs

def concretize_with_clustcc(specs: list[Spec]):
    with require_clustcc():
        to_concretize = [(s, None) for s in specs]
        if len(to_concretize) == 1:
            return [concretize_one(specs[0])]
        else:
            return [concr for _, concr in concretize_separately(to_concretize)]


