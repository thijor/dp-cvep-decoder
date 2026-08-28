"""Compilation smoke tests.

Byte-compilation does not import the third-party dependencies, so it runs in
any environment and guards against syntax errors across the whole codebase.
An additional import test runs only when the runtime deps are installed.
"""

import compileall
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _deps_available() -> bool:
    mods = ("pyntbci", "dareplane_utils", "pyxdf", "pylsl", "fire")
    return all(importlib.util.find_spec(m) is not None for m in mods)


HAVE_DEPS = _deps_available()


class TestCompilation(unittest.TestCase):
    def test_sources_byte_compile(self):
        """Every .py file under the package/api must byte-compile."""
        for target in ("cvep_decoder", "api"):
            with self.subTest(target=target):
                ok = compileall.compile_dir(
                    str(REPO / target), quiet=1, maxlevels=20, force=True
                )
                self.assertTrue(ok, f"Byte-compilation failed under {target}/")

    @unittest.skipUnless(HAVE_DEPS, "runtime dependencies not installed")
    def test_import_modules(self):
        """Importing additionally catches import-time errors (missing/renamed
        imports, bad relative imports), which byte-compilation cannot."""
        import cvep_decoder.online_decoding  # noqa: F401
        import cvep_decoder.train_decoder  # noqa: F401


if __name__ == "__main__":
    unittest.main()
