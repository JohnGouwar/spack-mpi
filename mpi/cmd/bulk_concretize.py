from argparse import ArgumentParser
from mpi.concretize import require_clustcc
from spack.spec import Spec
from pathlib import Path
try:
    from spack.extensions.mpi.concretize import require_clustcc, best_effort_concretize
    from spack.extensions.mpi.jsonl import write_specs_to_jsonl
except:
    from concretize import require_clustcc, best_effort_concretize
    from jsonl import write_specs_to_jsonl

level = "long"
description = "attempt to concretize many specs, dumping successes to JSON"
section = "concretize"


def setup_parser(parser: ArgumentParser):
    parser.add_argument(
        "--spec-file",
        required=True,
        type=Path,
        help="Newline separated .txt file of specs to attempt to concretize"
    )
    parser.add_argument(
        "--output-file",
        default=Path("concretized.jsonl"),
        type=Path,
        help="jsonl file to store concretization results (default: concretized.jsonl)"
    )
    parser.add_argument(
        "--add-clustcc",
        action="store_true",
        help="Add clustcc to concretized specs"
    )
                
def bulk_concretize(parser, args):
    spec_file : Path = args.spec_file
    output_file : Path = args.output_file
    assert spec_file.name.endswith(".txt"), "Input file must be a text file"
    assert output_file.name.endswith(".jsonl"), "Output must be in jsonl format"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(spec_file, "r") as f:
        specs = [Spec(l) for l in f]
    if args.add_clustcc:
        with require_clustcc():
            concretized = best_effort_concretize(specs)
    else:
        concretized = best_effort_concretize(specs)
    write_specs_to_jsonl(specs, concretized, output_file)
    
