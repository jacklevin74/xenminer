from ethapi.config import cli_args, BLOCKCHAIN_DB_URL
from sqlalchemy import create_engine, Connection


def connect() -> Connection:
    """
    Connects to the blockchain database
    :return: A database connection
    """
    return create_engine(
        BLOCKCHAIN_DB_URL,
        echo=cli_args().verbose,
        connect_args={"check_same_thread": False},
    ).connect()
