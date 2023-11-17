import logging
import gunicorn.app.base
from ethapi.server import app
from ethapi.config import cli_args

args = cli_args()
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(encoding="utf-8", level=log_level)


# pylint: disable=W0223
class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, application, options=None):
        self.options = options or {}
        self.application = application
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    if args.dev:
        app.run(host=args.host, port=args.port, debug=True, use_reloader=True)
    else:
        StandaloneApplication(app, {"bind": f"{args.host}:{args.port}"}).run()
