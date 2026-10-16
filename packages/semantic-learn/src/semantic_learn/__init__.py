from importlib import import_module
from importlib.metadata import version

__version__ = version('semantic-learn')
__all__ = ['collect', 'diagnose']
LAZY = {'collect': ('teacher', 'collect'), 'diagnose': ('review', 'diagnose')}


def __getattr__(name):
    if name not in LAZY:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = LAZY[name]
    value = globals()[name] = getattr(import_module(f'.{module}', __name__), attribute)
    return value
