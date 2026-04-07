

"""Compatibility wrapper for GRPO package.

This file calls into the refactored GRPO.main module. The original monolithic
script was split into submodules under `GRPO/` for clarity.
"""

from GRPO.models import main as _main


def main():
    _main.main()


if __name__ == '__main__':
    main()
