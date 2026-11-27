"""Write a tiny example checkpoint (model.safetensors) for the starter policy.

    python3 scripts/make_example_weights.py OUT_DIR
Upload OUT_DIR to a public Hugging Face model repo to practice the full flow with the starter.
Standard library only: no torch or safetensors needed. The file loads with
`safetensors.torch.load_file` as float32 weight [7, 6] and bias [7], all zeros.
"""
import json
import struct
import sys
from pathlib import Path


def write_example_weights(directory, scale: float = 0.0) -> Path:
    """Write weight [7, 6] and bias [7] as a float32 safetensors file with the standard library.

    Every weight is `scale` and every bias `0.01 * scale` (zeros by default). Returns the file path.
    """
    tensors = [("bias", [7], [0.01 * scale] * 7), ("weight", [7, 6], [scale] * 42)]
    header, blob = {}, b""
    for name, shape, values in tensors:
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path = Path(directory) / "model.safetensors"
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(head)) + head + blob)
    return path


def write(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_example_weights(out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(write(Path(sys.argv[1])))
