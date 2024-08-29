from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_cors import CORS

db = SQLAlchemy()
cors = CORS()
cache = Cache()
