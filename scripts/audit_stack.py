#!/usr/bin/env python3
"""Write the local stack manifest. GPU fields stay null until a host audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantic_locomotion.stack_audit import audit_local, to_yaml


def main() -> None:
    data = audit_local()
    out = ROOT / "config" / "stack.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_yaml(data))
    print(out)


if __name__ == "__main__":
    main()
