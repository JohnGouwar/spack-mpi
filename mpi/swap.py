from spack.spec import Spec
from spack.concretize import concretize_together, SpecPairInput
from typing import Optional
def _swap_in_spec(
        old_spec: Spec,
        mapping: dict[str, Spec],
        cache : Optional[dict[Spec, Spec]]= None
):
    '''
    Update the dependencies of `old_spec` by `mapping`. If provided cache is
    non-null, the swap is done transitively
    '''
    if cache and old_spec in cache:
        return cache[old_spec]
    swapped_spec = old_spec.copy(deps=False)
    swapped_spec.clear_caches(ignore=("package_hash",))
    for edge in old_spec.edges_to_dependencies():
        if edge.spec.name in mapping:
            spec_to_inject = mapping[edge.spec.name]
            swapped_spec.add_dependency_edge(
                spec_to_inject,
                depflag=edge.depflag,
                virtuals=edge.virtuals
            )
        else:
            if cache is not None:
                swapped_dep = _swap_in_spec(edge.spec, mapping, cache)
            else:
                swapped_dep = edge.spec
            swapped_spec.add_dependency_edge(
                swapped_dep,
                depflag=edge.depflag,
                virtuals=edge.virtuals
            )
    if cache is not None:
        cache[old_spec] = swapped_spec
    return swapped_spec


def concretize_with_clustcc(specs: list[Spec]) -> list[Spec]:
    '''
    Swap in the standard compiler wrapper for a clustcc variant
    '''
    to_concretize : list[SpecPairInput]= [(s, None) for s in specs]
    to_concretize.append((Spec("clustcc-compiler-wrapper"), None))
    concretized_all = concretize_together(to_concretize)
    concretized = []
    clustcc_wrapper = None
    for (user, concr) in concretized_all:
        if user.name == "clustcc-compiler-wrapper":
            clustcc_wrapper = concr
        else:
            concretized.append(concr)
    assert clustcc_wrapper is not None, "clustcc-compiler-wrapper should be among compiled specs"
    mapping = {"compiler-wrapper": clustcc_wrapper}
    new_specs = []
    cache = {}
    for s in concretized:
        new_specs.append(_swap_in_spec(s, mapping, cache))
    return new_specs
