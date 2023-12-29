from __future__ import annotations

from . import add, subtract

if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(
        {
            "add": add,
            "subtract": subtract,
        }
    )
