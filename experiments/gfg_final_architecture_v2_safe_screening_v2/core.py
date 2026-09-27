"""Fixed-shape FP32 primitive; reuse all original control-policy bytecode."""
from experiments.gfg_final_architecture_v2_safe_screening_v1.core import *
from experiments.gfg_final_architecture_v2_safe_screening_v1.core import Screen as PreviousScreen
from types import FunctionType, SimpleNamespace
import inspect

HERE = Path(__file__).resolve().parent
ROOT = Path('E:/gfg-evidence/final-v2-safe-screening-v2')
TILE = 256


@torch.no_grad()
def fixed_affine(state, h, seen=80):
    """Every matmul has exactly 256 rows, including a zero-padded final tile.

    This defines one numerical authority shared by full and sparse checking.
    It is not claimed bit-identical to the historical variable-shape cuBLAS.
    """
    if not len(h):
        return h.new_empty((0, seen))
    parts = []
    for start in range(0, len(h), TILE):
        count = min(TILE, len(h)-start)
        block = h.new_zeros((TILE, h.shape[1]))
        block[:count].copy_(h[start:start+count])
        z = nn.functional.linear(block, state['weight'], state['bias'])
        parts.append(z[:count, :seen])
    return torch.cat(parts, 0)


def with_affine(fn):
    """Local dependency injection; no mutation of imported module globals."""
    original = inspect.unwrap(fn)
    namespace = dict(original.__globals__)
    proxy = dict(vars(namespace['p']))
    proxy['affine'] = fixed_affine
    namespace['p'] = SimpleNamespace(**proxy)
    cloned = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    return torch.no_grad()(cloned)


class Screen(PreviousScreen):
    evaluate = with_affine(PreviousScreen.evaluate)


# Unchanged production control algorithm, changing ONLY its affine primitive.
reference_projection = with_affine(floor_projection)


def source_hashes():
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'experiments'],cwd=REPO,text=True).splitlines()
    files = [REPO/q for q in names if q.endswith('.py')]
    first = HERE.parent/'gfg_final_architecture_v2_safe_screening_v1'
    files += list(first.glob('*.py'))+[first/'PROTOCOL.md']
    files += list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {q.relative_to(REPO).as_posix():sha(q) for q in sorted(set(files))}
