import os
from dotenv import load_dotenv
import logging

load_dotenv("../.env")

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", False)
    SQLALCHEMY_BINDS = {
        "blocks": os.getenv("SQL_BLOCKS_URL", "sqlite:///" + basedir + "/../blocks.db"),
        "cache": os.getenv("SQL_CACHE_URL", "sqlite:///" + basedir + "/../cache.db"),
        "difficulty": os.getenv(
            "SQL_DIFFICULTY_URL",
            "sqlite:///" + basedir + "/../difficulty.db",
        ),
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 30
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = os.getenv("FLASK_PORT", 5566)
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", True)
