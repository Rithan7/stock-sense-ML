import os
import json
import warnings
import datetime
import traceback

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as scipy_stats
import plotly.graph_objects as go
import plotly.utils

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler

from models import db, User, Prediction, Watchlist

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"]                     = os.getenv("SECRET_KEY", "dev-secret-key-change-this")
app.config["SQLALCHEMY_DATABASE_URI"]        = os.getenv("DATABASE_URL", "sqlite:///stocksense.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
bcrypt        = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


STOCKS = {
    "AAPL":  "Apple Inc.",
    "TSLA":  "Tesla Inc.",
    "GOOGL": "Alphabet (Google)",
    "MSFT":  "Microsoft Corp.",
    "NVDA":  "NVIDIA Corp.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms",
    "JPM":   "JPMorgan Chase",
    "V":     "Visa Inc.",
    "NFLX":  "Netflix Inc.",
}

FEATURES = [
    "Return", "MA5", "MA10", "MA20", "Vol_Change", "Volatility",
    "Price_Range", "MA_Signal", "Momentum", "RSI", "MACD", "MACD_Signal",
    "BB_Width", "BB_Pos",
]

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
    font=dict(color="#8b949e", size=11), margin=dict(l=50, r=20, t=45, b=40),
)
AXIS = dict(gridcolor="#30363d", linecolor="#30363d")


def cj(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) if isinstance(col, tuple) else str(col)
                      for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def build_features(df):
    d = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    
    if "Volume" in d.columns:
        d["Volume"] = d["Volume"].replace(0, np.nan).ffill().bfill().fillna(1.0)
    d.dropna(subset=["Close", "High", "Low", "Volume"], inplace=True)

    d["Return"]      = d["Close"].pct_change()
    d["MA5"]         = d["Close"].rolling(5).mean()
    d["MA10"]        = d["Close"].rolling(10).mean()
    d["MA20"]        = d["Close"].rolling(20).mean()
    d["Vol_Change"]  = d["Volume"].pct_change()
    d["Volatility"]  = d["Return"].rolling(5).std()
    d["Price_Range"] = (d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)
    d["MA_Signal"]   = (d["MA5"] - d["MA10"]) / d["Close"].replace(0, np.nan)
    d["Momentum"]    = d["Close"] - d["Close"].shift(5)

    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12            = d["Close"].ewm(span=12, adjust=False).mean()
    ema26            = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    bb_m          = d["Close"].rolling(20).mean()
    bb_s          = d["Close"].rolling(20).std()
    d["BB_Width"] = (bb_m + bb_s * 2 - (bb_m - bb_s * 2)) / bb_m.replace(0, np.nan)
    d["BB_Pos"]   = (d["Close"] - (bb_m - bb_s * 2)) / ((bb_s * 4) + 1e-9)

    d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)

    # Clean out infinities and missing values
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(subset=FEATURES + ["Target"], inplace=True)

    for f in FEATURES:
        d[f] = np.clip(d[f], -1e5, 1e5)
    return d


def train_model(model_type, Xtr, Xte, y_train, y_test):
    if model_type == "rf":
        m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        m = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    m.fit(Xtr, y_train)
    yp = np.array(m.predict_proba(Xte)[:, 1], dtype=np.float64)
    yt = np.array(y_test, dtype=np.int32)

    best_t = 0.5
    best_f = 0.0
    for t in [i / 100 for i in range(5, 96)]:
        f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    ypred = (yp >= best_t).astype(int)

    acc = float(accuracy_score(yt, ypred))
    fpr, tpr, _ = roc_curve(yt, yp)
    ra  = float(auc(fpr, tpr))
    cm  = confusion_matrix(yt, ypred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1v  = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return m, yp, yt, acc, ra, f1v, best_t, fpr, tpr, cm, tn, fp, fn, tp, prec, rec


def fetch_news(ticker, limit=5):
    news_list = []
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        for i, item in enumerate(raw_news[:limit]):
            content = item.get("content", {})
            title     = content.get("title", item.get("title", ""))
            publisher = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
            link      = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
            pub_time  = content.get("pubDate", "") or ""
            summary   = content.get("summary", "") or content.get("description", "") or item.get("summary", "") or title
            if pub_time:
                try:
                    dt = datetime.datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                    now = datetime.datetime.now(datetime.timezone.utc)
                    diff = now - dt
                    hours = int(diff.total_seconds() // 3600)
                    if hours < 1:
                        time_ago = "Just now"
                    elif hours < 24:
                        time_ago = f"{hours}h ago"
                    else:
                        time_ago = f"{hours // 24}d ago"
                except Exception:
                    time_ago = pub_time[:10] if len(pub_time) >= 10 else pub_time
            else:
                prov_time = item.get("providerPublishTime", 0)
                if prov_time:
                    dt = datetime.datetime.fromtimestamp(prov_time, tz=datetime.timezone.utc)
                    now = datetime.datetime.now(datetime.timezone.utc)
                    diff = now - dt
                    hours = int(diff.total_seconds() // 3600)
                    if hours < 1:
                        time_ago = "Just now"
                    elif hours < 24:
                        time_ago = f"{hours}h ago"
                    else:
                        time_ago = f"{hours // 24}d ago"
                else:
                    time_ago = "Recently"
            if title:
                news_list.append({
                    "id":        f"news-{i}",
                    "title":     title,
                    "publisher": publisher or "Financial News",
                    "link":      link,
                    "time":      time_ago,
                    "summary":   summary,
                })
    except Exception:
        pass
    return news_list


# ── Auth Routes ────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template("register.html")

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        user    = User(username=username, email=email, password_hash=pw_hash)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    return render_template("history.html")


@app.route("/search")
def search():
    return render_template("search.html")


# ── API Routes ─────────────────────────────────────────────────────────────

@app.route("/api/history")
@login_required
def api_history():
    preds = (Prediction.query
             .filter_by(user_id=current_user.id)
             .order_by(Prediction.created_at.desc())
             .limit(20).all())
    return jsonify([p.to_dict() for p in preds])


@app.route("/api/watchlist", methods=["GET"])
@login_required
def api_watchlist_get():
    items = Watchlist.query.filter_by(user_id=current_user.id).order_by(Watchlist.added_at.desc()).all()
    return jsonify([i.to_dict() for i in items])


@app.route("/api/watchlist", methods=["POST"])
@login_required
def api_watchlist_add():
    data   = request.get_json(force=True)
    ticker = str(data.get("ticker", "")).upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    existing = Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first()
    if existing:
        return jsonify({"message": "Already in watchlist"})
    item = Watchlist(user_id=current_user.id, ticker=ticker)
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Added", "item": item.to_dict()})


@app.route("/api/watchlist/<ticker>", methods=["DELETE"])
@login_required
def api_watchlist_remove(ticker):
    ticker = ticker.upper().strip()
    item   = Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({"message": "Removed"})


@app.route("/api/liveprice/<ticker>")
def api_liveprice(ticker):
    ticker = ticker.upper().strip()
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if hist.empty:
            return jsonify({"error": "No data"}), 404
        price  = float(hist["Close"].iloc[-1])
        prev   = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change = round(price - prev, 2)
        pct    = round((change / prev) * 100, 2) if prev else 0.0
        return jsonify({"ticker": ticker, "price": round(price, 2), "change": change, "pct": pct})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stockchart/<ticker>")
def api_stockchart(ticker):
    ticker = ticker.upper().strip()
    period = request.args.get("period", "1y").lower()
    period_map = {
        "1m": "1mo",
        "3m": "3mo",
        "6m": "6mo",
        "1y": "1y",
        "5y": "5y"
    }
    yf_period = period_map.get(period, "1y")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=yf_period, auto_adjust=True)
        if hist.empty:
            return jsonify({"error": "No chart data available"}), 404
        
        hist = flatten_columns(hist)
        dates = list(hist.index.strftime("%Y-%m-%d"))
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dates,
            open=list(hist["Open"].astype(float)),
            high=list(hist["High"].astype(float)),
            low=list(hist["Low"].astype(float)),
            close=list(hist["Close"].astype(float)),
            name="Price",
            increasing=dict(line=dict(color="#56d364"), fillcolor="rgba(86,211,100,0.3)"),
            decreasing=dict(line=dict(color="#f85149"), fillcolor="rgba(248,81,73,0.3)"),
        ))
        
        if len(hist) >= 5:
            ma20 = hist["Close"].rolling(min(20, len(hist))).mean()
            fig.add_trace(go.Scatter(x=dates, y=list(ma20), line=dict(color="#4fc3f7", width=1.2), name="MA20"))
        if len(hist) >= 15:
            ma50 = hist["Close"].rolling(min(50, len(hist))).mean()
            fig.add_trace(go.Scatter(x=dates, y=list(ma50), line=dict(color="#e3b341", width=1.2), name="MA50"))
            
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=11), margin=dict(l=50, r=20, t=50, b=40),
            xaxis=dict(gridcolor="#30363d", linecolor="#30363d", rangeslider=dict(visible=False)),
            yaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
            title=dict(
                text=f"{ticker} — Interactive Price Chart ({period.upper()})",
                font=dict(color="#e6edf3", size=13),
                x=0.01, y=0.98, xanchor="left", yanchor="top"
            ),
            legend=dict(
                bgcolor="rgba(13,17,23,0.7)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(color="#e6edf3", size=11),
                orientation="h",
                x=0.01, y=0.01, xanchor="left", yanchor="bottom"
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", family="DM Sans, sans-serif", size=11)
            ),
        )
        return jsonify({"chart": cj(fig)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stockinfo/<ticker>")
def api_stockinfo(ticker):
    ticker = ticker.upper().strip()
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        def safe(key, default="N/A"):
            val = info.get(key, default)
            if val is None or val == "":
                return default
            return val

        def fmt_currency(val):
            if val is None or not isinstance(val, (int, float)):
                return "N/A"
            if abs(val) >= 1e12:
                return f"${val/1e12:.2f}T"
            elif abs(val) >= 1e9:
                return f"${val/1e9:.2f}B"
            elif abs(val) >= 1e6:
                return f"${val/1e6:.2f}M"
            else:
                return f"${val:,.2f}"

        def fmt_num(val, dec=2, pct=False):
            if val is None or not isinstance(val, (int, float)):
                return "N/A"
            if pct:
                return f"{val * 100:.2f}%"
            return f"{val:.{dec}f}"

        desc = safe("longBusinessSummary", "No company summary available.")

        employees = safe("fullTimeEmployees", None)
        emp_str = f"{employees:,}" if employees and isinstance(employees, int) else "N/A"

        price_val = safe("currentPrice", None) or safe("regularMarketPrice", None)
        price_str = f"${price_val:.2f}" if price_val and isinstance(price_val, (int, float)) else "N/A"

        hi52 = safe("fiftyTwoWeekHigh", None)
        lo52 = safe("fiftyTwoWeekLow", None)
        hi52_str = f"${hi52:.2f}" if hi52 and isinstance(hi52, (int, float)) else "N/A"
        lo52_str = f"${lo52:.2f}" if lo52 and isinstance(lo52, (int, float)) else "N/A"

        news = fetch_news(ticker, 6)

        return jsonify({
            "ticker":        ticker,
            "name":          safe("longName", safe("shortName", ticker)),
            "sector":        safe("sector"),
            "industry":      safe("industry"),
            "exchange":      safe("exchange", safe("fullExchangeName")),
            "country":       safe("country"),
            "website":       safe("website"),
            "price":         price_str,
            "week52High":    hi52_str,
            "week52Low":     lo52_str,
            "marketCap":     fmt_currency(info.get("marketCap")),
            "pe":            fmt_num(info.get("trailingPE")),
            "forwardPE":     fmt_num(info.get("forwardPE")),
            "pegRatio":      fmt_num(info.get("pegRatio")),
            "priceToBook":   fmt_num(info.get("priceToBook")),
            "beta":          fmt_num(info.get("beta")),
            "divYield":      fmt_num(info.get("dividendYield"), pct=True),
            "profitMargins": fmt_num(info.get("profitMargins"), pct=True),
            "revenue":       fmt_currency(info.get("totalRevenue")),
            "totalCash":     fmt_currency(info.get("totalCash")),
            "totalDebt":     fmt_currency(info.get("totalDebt")),
            "employees":     emp_str,
            "description":   desc,
            "news":          news,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── ML Route ───────────────────────────────────────────────────────────────

@app.route("/run", methods=["POST"])
@login_required
def run_model():
    try:
        p          = request.get_json(force=True)
        ticker     = str(p.get("ticker", "AAPL")).upper().strip()
        model_type = str(p.get("model", "logistic")).strip()
        test_size  = float(p.get("test_size", 0.2))
        try:
            start_year = int(p.get("start", 2020))
        except (ValueError, TypeError):
            start_year = 2020

        start_date = f"{start_year}-01-01"
        end_date   = datetime.date.today().strftime("%Y-%m-%d")

        raw      = None
        last_err = ""
        for _ in range(3):
            try:
                raw = yf.download(
                    ticker, start=start_date, end=end_date,
                    auto_adjust=True, progress=False, timeout=15,
                )
                if raw is not None and len(raw) > 0:
                    break
                last_err = "Empty data returned"
            except Exception as e:
                last_err = str(e)

        if raw is None or len(raw) == 0:
            return jsonify({"error": f"Could not fetch data for {ticker}. ({last_err})"}), 400

        raw = flatten_columns(raw)
        if "Close" not in raw.columns and "Adj Close" in raw.columns:
            raw.rename(columns={"Adj Close": "Close"}, inplace=True)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in raw.columns:
                if c in ["Open", "High", "Low"]:
                    if c == "Open":
                        raw["Open"] = raw["Close"]
                    elif c == "High":
                        raw["High"] = raw["Close"]
                    elif c == "Low":
                        raw["Low"] = raw["Close"]
                else:
                    return jsonify({"error": f"Missing column '{c}'."}), 400
            if isinstance(raw[c], pd.DataFrame):
                raw[c] = raw[c].iloc[:, 0]

        if len(raw) < 60:
            return jsonify({"error": "Not enough data. Try an earlier start year."}), 400

        sdf = build_features(raw)
        X   = sdf[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y   = sdf["Target"]

        if y.nunique() < 2:
            return jsonify({"error": "All days moved in the same direction — not enough variability."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)

        if y_test.nunique() < 2:
            return jsonify({"error": "Test set has only one class. Try a larger test size."}), 400

        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)

        # Train selected model
        m, yp, yt, acc, ra, f1v, best_t, fpr, tpr, cm, tn, fp, fn, tp, prec, rec = train_model(
            model_type, Xtr, Xte, y_train, y_test)

        # Train other model for comparison (Feature 4)
        other_type = "rf" if model_type == "logistic" else "logistic"
        m2, yp2, yt2, acc2, ra2, f1v2, best_t2, fpr2, tpr2, cm2, tn2, fp2, fn2, tp2, prec2, rec2 = train_model(
            other_type, Xtr, Xte, y_train, y_test)

        comparison = {
            model_type: {
                "accuracy": round(acc * 100, 1),
                "auc":      round(ra, 3),
                "f1":       round(f1v * 100, 1),
            },
            other_type: {
                "accuracy": round(acc2 * 100, 1),
                "auc":      round(ra2, 3),
                "f1":       round(f1v2 * 100, 1),
            },
        }

        gc        = float(2 * ra - 1)
        spec      = tn / (tn + fp) if tn + fp > 0 else 0.0
        gini_label = "Good" if gc >= 0.2 else ("Weak" if gc >= 0 else "Below random")

        hl = 0.0
        for b in np.array_split(np.argsort(yp), 10):
            if len(b) == 0:
                continue
            o = float(yt[b].sum())
            e = float(yp[b].sum())
            n = float(len(b))
            if e > 0 and (n - e) > 0:
                hl += (o - e) ** 2 / e + ((n - o) - (n - e)) ** 2 / (n - e)
        hl_p = float(1 - scipy_stats.chi2.cdf(hl, df=8))

        pu   = float(m.predict_proba(sc.transform(sdf[FEATURES].iloc[[-1]]))[0][1])
        pdir = "UP" if pu >= best_t else "DOWN"

        if model_type == "rf":
            coefs = np.array(m.feature_importances_, dtype=np.float64)
            se    = coefs * 0.05 + 1e-6
        else:
            coefs = np.array(m.coef_[0], dtype=np.float64)
            try:
                ph = np.array(m.predict_proba(Xtr)[:, 1], dtype=np.float64)
                W  = ph * (1 - ph)
                se = np.sqrt(np.diag(
                    np.linalg.inv(Xtr.T @ (Xtr * W[:, None]) + np.eye(len(FEATURES)) * 1e-6)
                ))
            except Exception:
                se = np.abs(coefs) * 0.1 + 1e-6

        wald = []
        for feat_name, cv, sv in sorted(zip(FEATURES, coefs, se), key=lambda x: abs(x[1]), reverse=True):
            cv = float(cv)
            sv = float(sv)
            z  = cv / sv if sv else 0.0
            pv = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            wald.append({
                "feature":   feat_name,
                "coef":      round(cv, 4),
                "direction": "Positive" if cv > 0 else "Negative",
                "z":         round(z, 2),
                "p":         round(pv, 4),
                "sig":       bool(pv < 0.05),
            })

        # ── Charts ────────────────────────────────────────────────────────
        rx         = list(sdf.index.astype(str))
        cl         = list(sdf["Close"].astype(float))
        split_date = X_test.index[0].strftime("%Y-%m-%d")
        price_min  = float(sdf["Close"].min())
        price_max  = float(sdf["Close"].max())

        # Chart 1 — Candlestick with MAs (Feature 2)
        c1 = go.Figure()
        c1.add_trace(go.Candlestick(
            x=rx,
            open=list(sdf["Open"].astype(float)),
            high=list(sdf["High"].astype(float)),
            low=list(sdf["Low"].astype(float)),
            close=cl,
            name="OHLC",
            increasing=dict(line=dict(color="#56d364"), fillcolor="rgba(86,211,100,0.3)"),
            decreasing=dict(line=dict(color="#f85149"), fillcolor="rgba(248,81,73,0.3)"),
        ))
        for col, color, lbl in [("MA5","#4fc3f7","MA5"),("MA10","#e3b341","MA10"),("MA20","#bc8cff","MA20")]:
            c1.add_trace(go.Scatter(
                x=rx, y=list(sdf[col].astype(float)),
                line=dict(color=color, width=1.4), name=lbl, opacity=0.9))
        c1.add_trace(go.Scatter(
            x=[split_date, split_date], y=[price_min, price_max],
            mode="lines", line=dict(color="#ffffff", width=2, dash="dash"),
            name=f"Split ({split_date})"))
        c1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=11), margin=dict(l=50, r=20, t=50, b=40),
            xaxis=dict(gridcolor="#30363d", linecolor="#30363d", rangeslider=dict(visible=False)),
            yaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
            title=dict(
                text=f"{ticker} — Candlestick & Moving Averages | Split: {split_date}",
                font=dict(color="#e6edf3", size=12),
                x=0.01, y=0.98, xanchor="left", yanchor="top"
            ),
            legend=dict(
                bgcolor="rgba(13,17,23,0.7)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(color="#e6edf3", size=10),
                orientation="h",
                x=0.01, y=0.01, xanchor="left", yanchor="bottom"
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", family="DM Sans, sans-serif", size=11)
            ),
        )

        # Chart 2 — ROC
        c2 = go.Figure()
        c2.add_trace(go.Scatter(
            x=list(fpr), y=list(tpr),
            line=dict(color="#4fc3f7", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.1)", name=f"AUC={ra:.3f}"))
        c2.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color="#555", dash="dash", width=1.2), name="Random"))
        c2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=11), margin=dict(l=55, r=20, t=50, b=45),
            xaxis=dict(gridcolor="#30363d", linecolor="#30363d", title=dict(text="FPR", font=dict(color="#8b949e",size=11))),
            yaxis=dict(gridcolor="#30363d", linecolor="#30363d", title=dict(text="TPR", font=dict(color="#8b949e",size=11))),
            title=dict(
                text=f"ROC Curve | Gini={gc:.3f} ({gini_label})",
                font=dict(color="#e6edf3", size=12),
                x=0.01, y=0.98, xanchor="left", yanchor="top"
            ),
            legend=dict(
                bgcolor="rgba(13,17,23,0.7)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(color="#e6edf3", size=10),
                x=0.98, y=0.05, xanchor="right", yanchor="bottom"
            ),
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", family="DM Sans, sans-serif", size=11)
            ),
        )

        # Chart 3 — Confusion Matrix
        c3 = go.Figure(go.Heatmap(
            z=[[tn, fp],[fn, tp]],
            x=["Pred DOWN","Pred UP"], y=["Actual DOWN","Actual UP"],
            text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
            texttemplate="%{text}", textfont=dict(size=15, color="#e6edf3"),
            colorscale=[[0,"#1c2128"],[0.5,"#1e3a5f"],[1,"#1a56db"]], showscale=False))
        c3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=11), margin=dict(l=50, r=20, t=45, b=40),
            title=dict(text="Confusion Matrix", font=dict(color="#e6edf3", size=12)),
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", family="DM Sans, sans-serif", size=11)
            ),
        )

        # Chart 4 — Feature Importance
        fp2 = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)
        fn3 = [x[0] for x in fp2]
        fv  = [float(x[1]) for x in fp2]
        fc  = ["#56d364" if v > 0 else "#f85149" for v in fv]
        coef_title = "Feature Importances" if model_type == "rf" else "Feature Coefficients"
        c4 = go.Figure(go.Bar(
            x=fv, y=fn3, orientation="h",
            marker=dict(color=fc, opacity=0.88),
            text=[f"{v:+.3f}" for v in fv], textposition="outside",
            textfont=dict(color="#e6edf3", size=9)))
        c4.add_trace(go.Scatter(
            x=[0,0], y=[fn3[0], fn3[-1]], mode="lines",
            line=dict(color="#e6edf3", width=0.8, dash="dash"), showlegend=False))
        c4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font=dict(color="#8b949e", size=11),
            xaxis=dict(gridcolor="#30363d", linecolor="#30363d", title="Value"),
            yaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
            title=dict(text=coef_title, font=dict(color="#e6edf3", size=12)),
            margin=dict(l=110, r=70, t=45, b=40),
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", family="DM Sans, sans-serif", size=11)
            ),
        )

        # Fetch news (Feature 3)
        news = fetch_news(ticker, 5)

        # Save to DB
        pred_record = Prediction(
            user_id=current_user.id, ticker=ticker,
            model_type="Random Forest" if model_type == "rf" else "Logistic Regression",
            accuracy=round(acc * 100, 1), auc=round(ra, 3),
            f1=round(f1v * 100, 1), prediction_direction=pdir,
            prob_up=round(pu * 100, 1),
        )
        db.session.add(pred_record)
        db.session.commit()

        return jsonify({
            "metrics": {
                "accuracy":    round(acc * 100, 1),
                "auc":         round(ra, 3),
                "gini":        round(gc, 3),
                "gini_label":  gini_label,
                "f1":          round(f1v * 100, 1),
                "specificity": round(spec * 100, 1),
                "threshold":   round(best_t, 2),
                "hl_p":        round(hl_p, 4),
                "hl_ok":       bool(hl_p > 0.05),
                "prob_up":     round(pu * 100, 1),
                "prediction":  pdir,
                "confidence":  round(max(pu, 1 - pu) * 100, 1),
                "ticker":      ticker,
                "company":     STOCKS.get(ticker, ticker),
                "model_name":  "Random Forest" if model_type == "rf" else "Logistic Regression",
                "train_rows":  int(len(X_train)),
                "test_rows":   int(len(X_test)),
                "total_days":  int(len(sdf)),
                "split_date":  split_date,
            },
            "wald":       wald,
            "comparison": comparison,
            "news":       news,
            "charts": {
                "price": cj(c1), "roc": cj(c2), "cm": cj(c3), "coef": cj(c4),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)