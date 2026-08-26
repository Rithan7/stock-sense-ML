from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    predictions = db.relationship("Prediction", backref="user", lazy=True)
    watchlist   = db.relationship("Watchlist",  backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.username}>"


class Prediction(db.Model):
    __tablename__ = "predictions"

    id                   = db.Column(db.Integer, primary_key=True)
    user_id              = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    ticker               = db.Column(db.String(20), nullable=False)
    model_type           = db.Column(db.String(50), nullable=False)
    accuracy             = db.Column(db.Float)
    auc                  = db.Column(db.Float)
    f1                   = db.Column(db.Float)
    prediction_direction = db.Column(db.String(10))
    prob_up              = db.Column(db.Float)
    created_at           = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id":                   self.id,
            "ticker":               self.ticker,
            "model_type":           self.model_type,
            "accuracy":             self.accuracy,
            "auc":                  self.auc,
            "f1":                   self.f1,
            "prediction_direction": self.prediction_direction,
            "prob_up":              self.prob_up,
            "created_at":           self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class Watchlist(db.Model):
    __tablename__ = "watchlist"

    id       = db.Column(db.Integer, primary_key=True)
    user_id  = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    ticker   = db.Column(db.String(20), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id":       self.id,
            "ticker":   self.ticker,
            "added_at": self.added_at.strftime("%Y-%m-%d %H:%M"),
        }
