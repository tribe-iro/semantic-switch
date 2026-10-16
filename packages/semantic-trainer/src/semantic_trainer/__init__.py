from importlib import import_module
from importlib.metadata import version

__version__ = version('semantic-trainer')
__all__ = ['train_lexical', 'qualify', 'diagnose']
LAZY = {'train_lexical': ('lexical', 'train', 'lexical', ('sklearn', 'scipy')),
        'qualify': ('qualification', 'qualify', 'lora', ('torch', 'transformers', 'peft')),
        'diagnose': ('qualification', 'diagnostic', 'lora', ('torch', 'transformers', 'peft'))}


def __getattr__(name):
    if name not in LAZY:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute, extra, requires = LAZY[name]
    try:
        value = getattr(import_module(f'.{module}', __name__), attribute)
    except ModuleNotFoundError as error:
        if (error.name or '').partition('.')[0] not in requires:
            raise
        raise ImportError(f'semantic_trainer.{name} needs the {extra!r} extra: pip install "semantic-trainer[{extra}]"', name=error.name) from error
    globals()[name] = value
    return value
