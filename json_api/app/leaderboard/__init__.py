from flask import Blueprint

bp = Blueprint("leaderboard", __name__)

from json_api.app.leaderboard import routes
