"""Path configuration for the legacy heuristic-corpus detector scripts.

The legacy scripts (build_classical_corpus.py, gen_classical_pointclouds.py,
train_1900_encoder.py, train_1900_v2.py, probe_1900.py, blind_test.py,
stat_final.py, combine_final.py, combine_final_v2.py) use files from this
repository and, for corpus construction, two inputs that are not distributed
with it:

* the formula-graph node file ``master_nodes.jsonl``: pass ``--master PATH``
  or set ``EPOCH_MASTER_GRAPH``;
* the directory that contains ``gen_dataside.py``, the formula-to-point-cloud
  generator: pass ``--gen-dataside DIR`` or set ``EPOCH_GEN_DATASIDE_DIR``.

Intermediate inputs (``classical_pointclouds.npz``, ``encoder_1900.pt``,
``boyle_1662.json``, measurement tables) are read from ``data/`` unless
``EPOCH_LEGACY_DIR`` points elsewhere.  Files the scripts produce are written
to ``output/`` unless ``EPOCH_LEGACY_OUT`` points elsewhere, so that a rerun
does not overwrite the released copies in ``data/``.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EPOCH_LEGACY_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("EPOCH_LEGACY_OUT", ROOT / "output"))

# Prefix strings used by the legacy scripts, which build paths by concatenation.
SC = str(DATA_DIR) + os.sep
CORPUS_BUILDER = str(ENGINE / "build_classical_corpus.py")


def _arg(flag):
    """Value of ``flag PATH`` or ``flag=PATH`` on the command line, else None."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _external(flag, env, what):
    value = _arg(flag) or os.environ.get(env)
    script = Path(sys.argv[0]).name or "script"
    if not value:
        sys.exit(
            f"{script}: {what} is not part of this repository. "
            f"Pass {flag} PATH or set the environment variable {env}."
        )
    path = Path(value).expanduser()
    if not path.exists():
        sys.exit(f"{script}: {what} not found at {path} (from {flag} / {env}).")
    return path


def master_graph():
    """Path of the formula-graph node file (master_nodes.jsonl)."""
    return str(_external("--master", "EPOCH_MASTER_GRAPH",
                         "the formula-graph node file (master_nodes.jsonl)"))


def add_gen_dataside():
    """Put the directory holding gen_dataside.py on sys.path."""
    d = _external("--gen-dataside", "EPOCH_GEN_DATASIDE_DIR",
                  "the directory containing gen_dataside.py")
    sys.path.insert(0, str(d))


def add_repo_root():
    """Put the repository root on sys.path so that ``models.encoders`` imports."""
    sys.path.insert(0, str(ROOT))


def out(name):
    """Path for an output file in OUT_DIR (created if needed)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / name)
