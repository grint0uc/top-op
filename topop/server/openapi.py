"""Print the OpenAPI document to stdout (`make types` pipes it to openapi.json)."""

from __future__ import annotations

import json
import sys

from topop.server.app import app


def main() -> None:
    json.dump(app.openapi(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
