from spack.spec import Spec
import json
from pathlib import Path
def write_specs_to_jsonl(abstract_specs, concrete_specs, output_file):
    with open(output_file, "w") as f:
        for (abstract, concrete) in zip(abstract_specs, concrete_specs):
            abstract_dict = abstract.to_dict()
            concrete_dict = concrete.to_dict()
            output_json = json.dumps(
                {"abstract": abstract_dict, "concrete": concrete_dict}
            )
            f.write(output_json+"\n");

def read_specs_from_jsonl(input_file: Path) -> list[tuple[Spec, Spec]]:
    assert input_file.name.endswith(".jsonl")
    output = []
    with open(input_file, "r") as f:
        for line in f:
            metadata_json = json.loads(line)
            abstract = Spec.from_dict(metadata_json['abstract'])
            concrete = Spec.from_dict(metadata_json['concrete'])
            output.append((abstract, concrete))
    return output
