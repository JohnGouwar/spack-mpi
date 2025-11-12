from dataclasses import dataclass
from typing import Optional


def _is_source(s: str) -> bool:
    SOURCE_EXTENSIONS = [".c", ".cc", ".C", ".cpp", ".cxx", ".c++"]
    return any(s.endswith(ext) for ext in SOURCE_EXTENSIONS)


@dataclass
class ParsedCommand:
    source_file: str
    source_index: int
    output_file: Optional[str]
    output_index: Optional[int]


def parse_compile_command_list(args: list[str]) -> ParsedCommand:
    source_file = None
    source_index = -1
    output_file = None
    output_index = None
    for i, arg in enumerate(args):
        if _is_source(arg):
            if source_file:
                raise ValueError("Multiple source files detected")
            else:
                source_file = arg
                source_index = i
        elif arg == "-o":
            if output_file:
                raise ValueError("Multiple output files detected")
            else:
                output_index = i + 1
                output_file = args[output_index]
    if source_file is None:
        raise ValueError("Unable to find source file")
    return ParsedCommand(
        source_file=source_file,
        source_index=source_index,
        output_file=output_file,
        output_index=output_index,
    )
