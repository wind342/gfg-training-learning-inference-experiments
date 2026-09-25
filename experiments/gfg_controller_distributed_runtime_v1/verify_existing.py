"""Post-run read-only wrapper. Scientific checks are the presealed audit.py.

Do not regenerate or overwrite any receipt: recompute the frozen audit and
require every would-be output to equal its already archived JSON object.
"""
from unittest.mock import patch
from . import audit


def compare(path,value):
    assert audit.b.read(path)==value,f'archived audit differs: {path}'


if __name__=='__main__':
    with patch.object(audit.b,'write',compare):audit.main()
