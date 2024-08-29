from json_api.app.extensions import db


class Blocks(db.Model):
    __tablename__ = "blocks"
    __bind_key__ = "blocks"
    block_id = db.Column(db.Integer, primary_key=True)
