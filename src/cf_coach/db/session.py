"""The database connection.

An Engine holds a pool of connections to Postgres. Create one per process and reuse it.
A Session is one unit of work on top of it: you open it, run queries, and commit or roll
back. Usage:

    with Session(get_engine()) as session, session.begin():
        ...   # commits at the end of the block, or rolls back if anything raises
"""

import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

_engine = None


def get_engine() -> Engine:
    # Created on first use (same reason as cf_client._limiter): importing this module
    # doesn't need DATABASE_URL.
    global _engine
    if _engine is None:
        load_dotenv()
        _engine = create_engine(os.environ["DATABASE_URL"])
    return _engine
