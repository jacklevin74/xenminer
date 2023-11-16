from server import app
from config import args
import logging

log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(encoding="utf-8", level=log_level)


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.verbose)
