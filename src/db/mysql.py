from __future__ import annotations

import aiomysql

from src.config.settings import DATABASE_CONFIG


async def connect_once() -> aiomysql.Connection:
    return await aiomysql.connect(
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
        db=DATABASE_CONFIG["database"],
        autocommit=True,
        charset="utf8mb4",
    )


async def ping() -> bool:
    conn = await connect_once()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()
            return row is not None
    finally:
        conn.close()


