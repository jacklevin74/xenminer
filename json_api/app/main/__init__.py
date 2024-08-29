from flask import Blueprint

bp = Blueprint("main", __name__)

from json_api.app.main import routes
