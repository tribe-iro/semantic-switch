import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from semantic_switch import tools

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "sklearn", "scipy", "llama_cpp", "transformers", "peft")


@pytest.mark.parametrize("name", ["lock", "export", "run_test", "explain_test"])
def test_readme_documents_each_tool_signature(name):
    readme = (ROOT / "README.md").read_text()
    documented = re.search(rf"`tools\.{name}\((.*?)\)(?: -> \w+)?`", readme)
    assert documented, name
    actual = str(inspect.signature(getattr(tools, name))).split(" -> ")[0].strip("()").replace("'", '"').replace("str | None", "")
    assert [p.split("=")[0].split(":")[0].strip() for p in documented.group(1).split(",")] == \
        [p.split("=")[0].split(":")[0].strip() for p in actual.split(",")]


@pytest.mark.parametrize("package", ["semantic_trainer", "semantic_learn"])
def test_optional_packages_import_without_fitting_or_inference_dependencies(package):
    code = f"import sys, {package}; print(sorted(set(sys.modules) & {set(HEAVY)!r}))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


def test_lazy_exports_are_the_owning_callables():
    import semantic_learn
    import semantic_trainer
    from semantic_learn.teacher import collect
    assert semantic_learn.__all__ == ["collect", "diagnose"] and semantic_learn.collect is collect
    assert semantic_trainer.__all__ == ["train_lexical", "qualify", "diagnose"]
    from semantic_trainer.qualification import qualify, diagnostic
    from semantic_learn.review import diagnose
    assert semantic_trainer.qualify is qualify and semantic_trainer.diagnose is diagnostic and semantic_learn.diagnose is diagnose
    with pytest.raises(AttributeError):
        semantic_trainer.missing


def test_missing_lexical_extra_names_the_install(monkeypatch):
    import importlib
    import semantic_trainer
    monkeypatch.delitem(semantic_trainer.__dict__, "train_lexical", raising=False)
    real = importlib.import_module

    def absent(name, package=None):
        if name == ".lexical":
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real(name, package)
    monkeypatch.setattr(semantic_trainer, "import_module", absent)
    with pytest.raises(ImportError, match=r'pip install "semantic-trainer\[lexical\]"'):
        semantic_trainer.train_lexical
