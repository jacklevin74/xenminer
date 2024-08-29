from ..config import Config
from flask import Flask
from .extensions import db, cors, cache
from .main import bp as main_bp
from .leaderboard import bp as leaderboard_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask extensions here
    db.init_app(app)
    cors.init_app(app)
    cache.init_app(app)

    # Register blueprints here
    app.register_blueprint(main_bp)
    app.register_blueprint(leaderboard_bp)

    return app
