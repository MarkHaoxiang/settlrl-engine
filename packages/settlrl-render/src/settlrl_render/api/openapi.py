"""Print the OpenAPI schema, the wire contract's source of truth.

``npm run gen-api`` (frontend/) pipes this into ``openapi.json`` and the
generated TypeScript types; ``test_openapi_schema_is_committed`` pins the
committed copy against the live app, so schema drift fails CI until both are
regenerated together.
"""

import json

from settlrl_render.server import create_app


def main() -> None:
    print(json.dumps(create_app().openapi(), indent=1))


if __name__ == "__main__":
    main()
