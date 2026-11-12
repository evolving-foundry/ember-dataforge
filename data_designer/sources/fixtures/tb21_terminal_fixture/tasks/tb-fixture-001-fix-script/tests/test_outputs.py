import pathlib

fixed = pathlib.Path("scripts/sum.py").read_text(encoding="utf-8")
compile(fixed, "sum.py", "exec")
assert "+" in fixed, "the operator must be restored"
assert "return a + b" in fixed, "total(a, b) must return a + b"
