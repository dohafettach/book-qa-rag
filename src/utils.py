

import os


def validate_env() -> None:
    """
    Check that required environment variables are set before the server starts.
    Crashes immediately with a clear message rather than failing silently
    on the first API call.
    """
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"\n\nMissing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your API keys.\n"
            "See README.md for where to get them.\n"
        )
