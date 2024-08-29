from .app import create_app, Config


if __name__ == "__main__":
    app = create_app(Config)
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=True)
