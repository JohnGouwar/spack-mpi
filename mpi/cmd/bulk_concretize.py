from argparse import ArgumentParser
from typing import Optional
from spack.spec import Spec
from spack.solver.reuse import SpecFiltersFactory, SpecFilter
from pathlib import Path
import tempfile
import spack.store
try:
    from spack.extensions.mpi.concretize import require_clustcc, best_effort_concretize
    from spack.extensions.mpi.jsonl import write_specs_to_jsonl
except:
    from ..concretize import require_clustcc, best_effort_concretize
    from ..jsonl import write_specs_to_jsonl

level = "long"
description = "attempt to concretize many specs, dumping successes to JSON"
section = "concretize"


def setup_parser(parser: ArgumentParser):
    parser.add_argument(
        "--spec-files",
        required=True,
        type=str,
        help="Comma separated list of newline separated .txt files of specs to "
        "attempt to concretize"
    )
    parser.add_argument(
        "--output-file",
        default=Path("concretized.jsonl"),
        type=Path,
        help="jsonl file to store concretization results (default: concretized.jsonl)"
    )
    parser.add_argument(
        "--add-clustcc",
        type=str,
        help="Add clustcc to concretized specs, provide the clustcc spec"
    )
    parser.add_argument(
        "--empty-store",
        action="store_true",
        help="Concretize in an empty store"
    )

def _concretize(specs, add_clustcc: Optional[str], already_concretized: list[Spec]):
    if add_clustcc:
        with require_clustcc(add_clustcc):
            concretized = best_effort_concretize(specs, already_concretized)
    else:
        concretized = best_effort_concretize(specs, already_concretized)
    return concretized

                
def bulk_concretize(parser, args):
    output_file : Path = args.output_file
    assert output_file.name.endswith(".jsonl"), "Output must be in jsonl format"
    output_file.parent.mkdir(exist_ok=True, parents=True)

    user_specs = []
    already_concretized = []
    for spec_file in args.spec_files.split(","):
        with open(spec_file, "r") as f:
            specs = [Spec(l) for l in f]
            user_specs += specs
        if args.empty_store:
            with tempfile.TemporaryDirectory() as td:
                with spack.store.use_store(td):
                    newly_concretized = _concretize(specs, args.add_clustcc, already_concretized)
        else:
            newly_concretized = _concretize(specs, args.add_clustcc, already_concretized)
        already_concretized += newly_concretized
            
    write_specs_to_jsonl(user_specs, already_concretized, output_file)
    
