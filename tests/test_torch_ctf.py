import torch_ctf


def test_imports_with_version():
    assert isinstance(torch_ctf.__version__, str)
