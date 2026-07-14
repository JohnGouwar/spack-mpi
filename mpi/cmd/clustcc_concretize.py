from ..concretize import concretize_with_clustcc
from spack.cmd.common import arguments 
from spack.spec import Spec
from spack.cmd import parse_specs
from argparse import ArgumentParser
level = "long"
description = "concretize a single spec with clustcc, mostly for debugging"
section = "concretize"

def setup_parser(parser: ArgumentParser):
    arguments.add_common_arguments(parser, ["spec"])
    parser.add_argument(
        "--clustcc-gcc-spec",
        type=Spec,
        default=Spec("clustcc-gcc"),
        help="the clustcc-gcc spec to concretize against"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump spec output as JSON"
    )

def clustcc_concretize(parser, args):
    concrete = concretize_with_clustcc(parse_specs(args.spec), args.clustcc_gcc_spec)
    if args.json:
        print(concrete[0].to_json(), end="")
    else:
        print(concrete[0].tree())
