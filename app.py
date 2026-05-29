import base64
import io
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


THEORETICAL_PARAMS = {
    "fast_ema": 12,
    "slow_ema": 26,
    "signal_ema": 9,
    "rsi_period": 14,
    "rsi_entry_floor": 50,
    "rsi_overheat_exit": 75,
}

TREND_DATASETS = {
    "台泥": {
        "ticker": "1101.TW",
        "trend_name": "台泥",
        "csv_b64": "IlRpbWUiLCLlj7Dms6UiCiIyMDE2LTA0LTAxIiw3CiIyMDE2LTA1LTAxIiw5CiIyMDE2LTA2LTAxIiw3CiIyMDE2LTA3LTAxIiw2CiIyMDE2LTA4LTAxIiw2CiIyMDE2LTA5LTAxIiw2CiIyMDE2LTEwLTAxIiw1CiIyMDE2LTExLTAxIiw1CiIyMDE2LTEyLTAxIiw1CiIyMDE3LTAxLTAxIiwxNgoiMjAxNy0wMi0wMSIsOAoiMjAxNy0wMy0wMSIsOAoiMjAxNy0wNC0wMSIsMTEKIjIwMTctMDUtMDEiLDcKIjIwMTctMDYtMDEiLDkKIjIwMTctMDctMDEiLDcKIjIwMTctMDgtMDEiLDgKIjIwMTctMDktMDEiLDcKIjIwMTctMTAtMDEiLDYKIjIwMTctMTEtMDEiLDcKIjIwMTctMTItMDEiLDE3CiIyMDE4LTAxLTAxIiwxMAoiMjAxOC0wMi0wMSIsNgoiMjAxOC0wMy0wMSIsMTAKIjIwMTgtMDQtMDEiLDgKIjIwMTgtMDUtMDEiLDExCiIyMDE4LTA2LTAxIiwyMgoiMjAxOC0wNy0wMSIsMTEKIjIwMTgtMDgtMDEiLDE0CiIyMDE4LTA5LTAxIiwxMQoiMjAxOC0xMC0wMSIsMTEKIjIwMTgtMTEtMDEiLDIwCiIyMDE4LTEyLTAxIiwxOQoiMjAxOS0wMS0wMSIsMTIKIjIwMTktMDItMDEiLDEyCiIyMDE5LTAzLTAxIiwyNQoiMjAxOS0wNC0wMSIsMjUKIjIwMTktMDUtMDEiLDMwCiIyMDE5LTA2LTAxIiwyNgoiMjAxOS0wNy0wMSIsMzMKIjIwMTktMDgtMDEiLDMwCiIyMDE5LTA5LTAxIiwyNQoiMjAxOS0xMC0wMSIsMjUKIjIwMTktMTEtMDEiLDI4CiIyMDE5LTEyLTAxIiwyNAoiMjAyMC0wMS0wMSIsMjcKIjIwMjAtMDItMDEiLDI2CiIyMDIwLTAzLTAxIiwzNAoiMjAyMC0wNC0wMSIsMjgKIjIwMjAtMDUtMDEiLDI5CiIyMDIwLTA2LTAxIiwzMAoiMjAyMC0wNy0wMSIsMzkKIjIwMjAtMDgtMDEiLDQyCiIyMDIwLTA5LTAxIiwzNgoiMjAyMC0xMC0wMSIsMjYKIjIwMjAtMTEtMDEiLDMxCiIyMDIwLTEyLTAxIiwzNwoiMjAyMS0wMS0wMSIsMzYKIjIwMjEtMDItMDEiLDM4CiIyMDIxLTAzLTAxIiw3MQoiMjAyMS0wNC0wMSIsMTAwCiIyMDIxLTA1LTAxIiw5MgoiMjAyMS0wNi0wMSIsNDkKIjIwMjEtMDctMDEiLDYyCiIyMDIxLTA4LTAxIiw2NgoiMjAyMS0wOS0wMSIsNTEKIjIwMjEtMTAtMDEiLDQ0CiIyMDIxLTExLTAxIiw0NwoiMjAyMS0xMi0wMSIsNDEKIjIwMjItMDEtMDEiLDMyCiIyMDIyLTAyLTAxIiwzNQoiMjAyMi0wMy0wMSIsNDYKIjIwMjItMDQtMDEiLDQ5CiIyMDIyLTA1LTAxIiw0NAoiMjAyMi0wNi0wMSIsMzgKIjIwMjItMDctMDEiLDUwCiIyMDIyLTA4LTAxIiw0OQoiMjAyMi0wOS0wMSIsNDEKIjIwMjItMTAtMDEiLDQ0CiIyMDIyLTExLTAxIiw1MwoiMjAyMi0xMi0wMSIsNDAKIjIwMjMtMDEtMDEiLDM0CiIyMDIzLTAyLTAxIiw0OQoiMjAyMy0wMy0wMSIsNTgKIjIwMjMtMDQtMDEiLDQ3CiIyMDIzLTA1LTAxIiw2OQoiMjAyMy0wNi0wMSIsNDMKIjIwMjMtMDctMDEiLDQ3CiIyMDIzLTA4LTAxIiwzNwoiMjAyMy0wOS0wMSIsNDAKIjIwMjMtMTAtMDEiLDM3CiIyMDIzLTExLTAxIiw0NQoiMjAyMy0xMi0wMSIsMzgKIjIwMjQtMDEtMDEiLDM0CiIyMDI0LTAyLTAxIiwzMQoiMjAyNC0wMy0wMSIsNDEKIjIwMjQtMDQtMDEiLDQyCiIyMDI0LTA1LTAxIiw0MwoiMjAyNC0wNi0wMSIsMzYKIjIwMjQtMDctMDEiLDM1CiIyMDI0LTA4LTAxIiwzMQoiMjAyNC0wOS0wMSIsMjIKIjIwMjQtMTAtMDEiLDE5CiIyMDI0LTExLTAxIiwyNAoiMjAyNC0xMi0wMSIsMjIKIjIwMjUtMDEtMDEiLDIwCiIyMDI1LTAyLTAxIiwyOAoiMjAyNS0wMy0wMSIsMzAKIjIwMjUtMDQtMDEiLDI2CiIyMDI1LTA1LTAxIiwzMgoiMjAyNS0wNi0wMSIsMzQKIjIwMjUtMDctMDEiLDM2CiIyMDI1LTA4LTAxIiw0NAoiMjAyNS0wOS0wMSIsMzUKIjIwMjUtMTAtMDEiLDI2CiIyMDI1LTExLTAxIiwyOQoiMjAyNS0xMi0wMSIsMjcKIjIwMjYtMDEtMDEiLDM1CiIyMDI2LTAyLTAxIiwyMQoiMjAyNi0wMy0wMSIsMzIKIjIwMjYtMDQtMDEiLDQ0CiIyMDI2LTA1LTAxIiw0Mw==",
    },
    "台積電": {
        "ticker": "2330.TW",
        "trend_name": "台積電",
        "csv_b64": "IlRpbWUiLCLlj7DnqY3pm7siCiIyMDE2LTA0LTAxIiw2CiIyMDE2LTA1LTAxIiw3CiIyMDE2LTA2LTAxIiw2CiIyMDE2LTA3LTAxIiw2CiIyMDE2LTA4LTAxIiw2CiIyMDE2LTA5LTAxIiw2CiIyMDE2LTEwLTAxIiw4CiIyMDE2LTExLTAxIiw3CiIyMDE2LTEyLTAxIiw3CiIyMDE3LTAxLTAxIiw2CiIyMDE3LTAyLTAxIiw3CiIyMDE3LTAzLTAxIiw4CiIyMDE3LTA0LTAxIiw3CiIyMDE3LTA1LTAxIiw3CiIyMDE3LTA2LTAxIiwxMAoiMjAxNy0wNy0wMSIsOAoiMjAxNy0wOC0wMSIsOAoiMjAxNy0wOS0wMSIsOAoiMjAxNy0xMC0wMSIsMTEKIjIwMTctMTEtMDEiLDEwCiIyMDE3LTEyLTAxIiw4CiIyMDE4LTAxLTAxIiwxMQoiMjAxOC0wMi0wMSIsOQoiMjAxOC0wMy0wMSIsMTEKIjIwMTgtMDQtMDEiLDEyCiIyMDE4LTA1LTAxIiwxMAoiMjAxOC0wNi0wMSIsMTEKIjIwMTgtMDctMDEiLDEwCiIyMDE4LTA4LTAxIiwxNAoiMjAxOC0wOS0wMSIsMTIKIjIwMTgtMTAtMDEiLDE1CiIyMDE4LTExLTAxIiwxMwoiMjAxOC0xMi0wMSIsMTEKIjIwMTktMDEtMDEiLDEzCiIyMDE5LTAyLTAxIiwxMAoiMjAxOS0wMy0wMSIsMTIKIjIwMTktMDQtMDEiLDEyCiIyMDE5LTA1LTAxIiwxNQoiMjAxOS0wNi0wMSIsMTQKIjIwMTktMDctMDEiLDE2CiIyMDE5LTA4LTAxIiwxNQoiMjAxOS0wOS0wMSIsMTUKIjIwMTktMTAtMDEiLDIwCiIyMDE5LTExLTAxIiwyMQoiMjAxOS0xMi0wMSIsMjQKIjIwMjAtMDEtMDEiLDIxCiIyMDIwLTAyLTAxIiwyMwoiMjAyMC0wMy0wMSIsNTAKIjIwMjAtMDQtMDEiLDM1CiIyMDIwLTA1LTAxIiwzNAoiMjAyMC0wNi0wMSIsMzIKIjIwMjAtMDctMDEiLDU4CiIyMDIwLTA4LTAxIiw1MQoiMjAyMC0wOS0wMSIsNDcKIjIwMjAtMTAtMDEiLDQyCiIyMDIwLTExLTAxIiw1MwoiMjAyMC0xMi0wMSIsNTYKIjIwMjEtMDEtMDEiLDEwMAoiMjAyMS0wMi0wMSIsNzgKIjIwMjEtMDMtMDEiLDk2CiIyMDIxLTA0LTAxIiw3MQoiMjAyMS0wNS0wMSIsNzIKIjIwMjEtMDYtMDEiLDUzCiIyMDIxLTA3LTAxIiw1MwoiMjAyMS0wOC0wMSIsNDgKIjIwMjEtMDktMDEiLDUzCiIyMDIxLTEwLTAxIiw0NwoiMjAyMS0xMS0wMSIsNDYKIjIwMjEtMTItMDEiLDQzCiIyMDIyLTAxLTAxIiw1MgoiMjAyMi0wMi0wMSIsNDYKIjIwMjItMDMtMDEiLDczCiIyMDIyLTA0LTAxIiw1NwoiMjAyMi0wNS0wMSIsNTQKIjIwMjItMDYtMDEiLDYwCiIyMDIyLTA3LTAxIiw2NwoiMjAyMi0wOC0wMSIsNTIKIjIwMjItMDktMDEiLDU0CiIyMDIyLTEwLTAxIiw3NwoiMjAyMi0xMS0wMSIsNzkKIjIwMjItMTItMDEiLDU1CiIyMDIzLTAxLTAxIiw0MgoiMjAyMy0wMi0wMSIsNTIKIjIwMjMtMDMtMDEiLDQyCiIyMDIzLTA0LTAxIiwzNgoiMjAyMy0wNS0wMSIsMzkKIjIwMjMtMDYtMDEiLDM4CiIyMDIzLTA3LTAxIiwzNgoiMjAyMy0wOC0wMSIsMzIKIjIwMjMtMDktMDEiLDMxCiIyMDIzLTEwLTAxIiwzMgoiMjAyMy0xMS0wMSIsMzQKIjIwMjMtMTItMDEiLDMxCiIyMDI0LTAxLTAxIiw0MAoiMjAyNC0wMi0wMSIsMzkKIjIwMjQtMDMtMDEiLDUzCiIyMDI0LTA0LTAxIiw1MQoiMjAyNC0wNS0wMSIsNDQKIjIwMjQtMDYtMDEiLDU3CiIyMDI0LTA3LTAxIiw3NQoiMjAyNC0wOC0wMSIsNjgKIjIwMjQtMDktMDEiLDUxCiIyMDI0LTEwLTAxIiw1NgoiMjAyNC0xMS0wMSIsNTIKIjIwMjQtMTItMDEiLDUxCiIyMDI1LTAxLTAxIiw1MAoiMjAyNS0wMi0wMSIsNTQKIjIwMjUtMDMtMDEiLDczCiIyMDI1LTA0LTAxIiw4OAoiMjAyNS0wNS0wMSIsNTAKIjIwMjUtMDYtMDEiLDQ0CiIyMDI1LTA3LTAxIiw0OAoiMjAyNS0wOC0wMSIsNTYKIjIwMjUtMDktMDEiLDQ5CiIyMDI1LTEwLTAxIiw1OQoiMjAyNS0xMS0wMSIsNTEKIjIwMjUtMTItMDEiLDQ2CiIyMDI2LTAxLTAxIiw3NAoiMjAyNi0wMi0wMSIsNjAKIjIwMjYtMDMtMDEiLDgwCiIyMDI2LTA0LTAxIiw4NwoiMjAyNi0wNS0wMSIsODU=",
    },
    "元大高股息 0056": {
        "ticker": "0056.TW",
        "trend_name": "0056",
        "csv_b64": "IlRpbWUiLCIwMDU2Iiwi5YWD5aSn6auY6IKh5oGvIgoiMjAxNi0wNC0wMSIsNSwwCiIyMDE2LTA1LTAxIiw4LDAKIjIwMTYtMDYtMDEiLDYsMAoiMjAxNi0wNy0wMSIsNSwwCiIyMDE2LTA4LTAxIiw0LDAKIjIwMTYtMDktMDEiLDUsMAoiMjAxNi0xMC0wMSIsNSwwCiIyMDE2LTExLTAxIiw4LDEKIjIwMTYtMTItMDEiLDUsMAoiMjAxNy0wMS0wMSIsNCwwCiIyMDE3LTAyLTAxIiw1LDEKIjIwMTctMDMtMDEiLDYsMAoiMjAxNy0wNC0wMSIsNSwwCiIyMDE3LTA1LTAxIiw1LDAKIjIwMTctMDYtMDEiLDUsMAoiMjAxNy0wNy0wMSIsNywxCiIyMDE3LTA4LTAxIiw2LDEKIjIwMTctMDktMDEiLDcsMQoiMjAxNy0xMC0wMSIsNywxCiIyMDE3LTExLTAxIiw2LDEKIjIwMTctMTItMDEiLDUsMAoiMjAxOC0wMS0wMSIsNiwxCiIyMDE4LTAyLTAxIiw2LDEKIjIwMTgtMDMtMDEiLDcsMQoiMjAxOC0wNC0wMSIsNSwwCiIyMDE4LTA1LTAxIiw2LDAKIjIwMTgtMDYtMDEiLDUsMAoiMjAxOC0wNy0wMSIsNywxCiIyMDE4LTA4LTAxIiw3LDAKIjIwMTgtMDktMDEiLDksMQoiMjAxOC0xMC0wMSIsMjEsMgoiMjAxOC0xMS0wMSIsMTQsMQoiMjAxOC0xMi0wMSIsMTMsMQoiMjAxOS0wMS0wMSIsMTUsMQoiMjAxOS0wMi0wMSIsMTUsMQoiMjAxOS0wMy0wMSIsMjAsMQoiMjAxOS0wNC0wMSIsMTcsMQoiMjAxOS0wNS0wMSIsMjUsMgoiMjAxOS0wNi0wMSIsMjIsMQoiMjAxOS0wNy0wMSIsMjAsMQoiMjAxOS0wOC0wMSIsMjUsMgoiMjAxOS0wOS0wMSIsMjMsMgoiMjAxOS0xMC0wMSIsMzYsMwoiMjAxOS0xMS0wMSIsMjgsMgoiMjAxOS0xMi0wMSIsMjMsMgoiMjAyMC0wMS0wMSIsMjIsMgoiMjAyMC0wMi0wMSIsMzEsNQoiMjAyMC0wMy0wMSIsODEsMTEKIjIwMjAtMDQtMDEiLDQ5LDYKIjIwMjAtMDUtMDEiLDQxLDUKIjIwMjAtMDYtMDEiLDQyLDYKIjIwMjAtMDctMDEiLDM5LDcKIjIwMjAtMDgtMDEiLDMzLDcKIjIwMjAtMDktMDEiLDMxLDgKIjIwMjAtMTAtMDEiLDQ3LDEwCiIyMDIwLTExLTAxIiw0Nyw4CiIyMDIwLTEyLTAxIiwzOCw3CiIyMDIxLTAxLTAxIiwzNCw5CiIyMDIxLTAyLTAxIiwzMiw4CiIyMDIxLTAzLTAxIiwzOSw4CiIyMDIxLTA0LTAxIiwzMCw5CiIyMDIxLTA1LTAxIiwzNSw3CiIyMDIxLTA2LTAxIiwyNiw1CiIyMDIxLTA3LTAxIiwzNSw2CiIyMDIxLTA4LTAxIiwzOCw1CiIyMDIxLTA5LTAxIiwzMiw1CiIyMDIxLTEwLTAxIiw1Nyw3CiIyMDIxLTExLTAxIiw0Nyw3CiIyMDIxLTEyLTAxIiwzNyw3CiIyMDIyLTAxLTAxIiwzMiw2CiIyMDIyLTAyLTAxIiw0MSw2CiIyMDIyLTAzLTAxIiw0Niw2CiIyMDIyLTA0LTAxIiwzNSw0CiIyMDIyLTA1LTAxIiwzMyw0CiIyMDIyLTA2LTAxIiw0Miw0CiIyMDIyLTA3LTAxIiw0OSw0CiIyMDIyLTA4LTAxIiw0Miw1CiIyMDIyLTA5LTAxIiw0Nyw0CiIyMDIyLTEwLTAxIiw4Myw3CiIyMDIyLTExLTAxIiw1NSw1CiIyMDIyLTEyLTAxIiw0Niw0CiIyMDIzLTAxLTAxIiwzNCw0CiIyMDIzLTAyLTAxIiw0Nyw0CiIyMDIzLTAzLTAxIiw0Niw0CiIyMDIzLTA0LTAxIiwzNSwzCiIyMDIzLTA1LTAxIiw0NCw0CiIyMDIzLTA2LTAxIiw1NSw2CiIyMDIzLTA3LTAxIiw3Niw3CiIyMDIzLTA4LTAxIiw1Nyw1CiIyMDIzLTA5LTAxIiw0Myw0CiIyMDIzLTEwLTAxIiw2MCw0CiIyMDIzLTExLTAxIiw1NCw1CiIyMDIzLTEyLTAxIiw1Niw1CiIyMDI0LTAxLTAxIiw2Niw1CiIyMDI0LTAyLTAxIiw0Niw0CiIyMDI0LTAzLTAxIiw1OCw1CiIyMDI0LTA0LTAxIiw1OCw0CiIyMDI0LTA1LTAxIiw0NiwzCiIyMDI0LTA2LTAxIiw0NSwzCiIyMDI0LTA3LTAxIiw1NCwzCiIyMDI0LTA4LTAxIiw1MSwzCiIyMDI0LTA5LTAxIiw0MiwyCiIyMDI0LTEwLTAxIiw1MCwzCiIyMDI0LTExLTAxIiw0MywyCiIyMDI0LTEyLTAxIiw1MiwzCiIyMDI1LTAxLTAxIiw1MywzCiIyMDI1LTAyLTAxIiw1OCwzCiIyMDI1LTAzLTAxIiw1NCwyCiIyMDI1LTA0LTAxIiw5MiwzCiIyMDI1LTA1LTAxIiw0OSwzCiIyMDI1LTA2LTAxIiw1NSwyCiIyMDI1LTA3LTAxIiw1OSwzCiIyMDI1LTA4LTAxIiw0NCwyCiIyMDI1LTA5LTAxIiw1MCwyCiIyMDI1LTEwLTAxIiw2NCwzCiIyMDI1LTExLTAxIiw1NSwyCiIyMDI1LTEyLTAxIiw1NSwzCiIyMDI2LTAxLTAxIiw3Niw0CiIyMDI2LTAyLTAxIiw1MywzCiIyMDI2LTAzLTAxIiw3MywzCiIyMDI2LTA0LTAxIiwxMDAsNAoiMjAyNi0wNS0wMSIsNzgsNA==",
    },
}


@dataclass
class BacktestMetrics:
    total_return: float
    buy_hold_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    trades: int
    final_equity: float


def parse_trends_csv(raw_csv: bytes, trend_name: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_csv)).dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty or len(df.columns) < 2:
        raise ValueError("Google Trends 資料格式不正確。")

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    df = df.dropna(subset=["date"]).sort_values("date")

    value_columns = [column for column in df.columns if column != "date"]
    for column in value_columns:
        cleaned = (
            df[column]
            .astype(str)
            .str.replace("<1", "0.5", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        df[column] = pd.to_numeric(cleaned, errors="coerce")

    selected_column = value_columns[0]
    return df[["date", selected_column]].rename(columns={selected_column: trend_name})


@st.cache_data
def load_trends_dataset(stock_name: str) -> pd.DataFrame:
    dataset = TREND_DATASETS[stock_name]
    raw_csv = base64.b64decode(dataset["csv_b64"])
    return parse_trends_csv(raw_csv, dataset["trend_name"])


@st.cache_data(ttl=60 * 30)
def fetch_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError("Yahoo Finance 沒有回傳資料，請確認股票代碼與日期區間。")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"adj_close": "close"})
    keep_columns = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in data]
    data = data[keep_columns].dropna(subset=["date", "close"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    if len(data) < 60:
        raise ValueError("資料筆數不足，請拉長回測期間。")
    return data


def add_indicators(
    df: pd.DataFrame,
    fast_ema: int,
    slow_ema: int,
    signal_ema: int,
    rsi_period: int,
    rsi_entry_floor: int,
    rsi_overheat_exit: int,
    trend_df: pd.DataFrame | None = None,
    trend_column: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    close = result["close"]

    ema_fast = close.ewm(span=fast_ema, adjust=False).mean()
    ema_slow = close.ewm(span=slow_ema, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal_ema, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["macd_bullish"] = result["macd"] > result["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    result["rsi_bullish"] = (result["rsi"] >= rsi_entry_floor) & (result["rsi"] < rsi_overheat_exit)

    if trend_df is not None and trend_column:
        trend = trend_df[["date", trend_column]].rename(columns={trend_column: "trend_interest"}).dropna()
        trend["date"] = pd.to_datetime(trend["date"], errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
        trend["trend_ma"] = trend["trend_interest"].rolling(4, min_periods=1).mean()
        merged = pd.merge_asof(result.sort_values("date"), trend.sort_values("date"), on="date", direction="backward")
        result["trend_interest"] = merged["trend_interest"]
        result["trend_ma"] = merged["trend_ma"]
        result["trend_bullish"] = merged["trend_interest"] >= merged["trend_ma"]
    else:
        result["trend_interest"] = np.nan
        result["trend_ma"] = np.nan
        result["trend_bullish"] = False

    return result


def build_combined_signal(df: pd.DataFrame, use_trend: bool, use_macd: bool, use_rsi: bool) -> pd.Series:
    signals = []
    if use_trend:
        signals.append(df["trend_bullish"].fillna(False))
    if use_macd:
        signals.append(df["macd_bullish"].fillna(False))
    if use_rsi:
        signals.append(df["rsi_bullish"].fillna(False))
    if not signals:
        return pd.Series(False, index=df.index)
    combined = signals[0].copy()
    for signal in signals[1:]:
        combined = combined & signal
    return combined


def run_backtest(
    df: pd.DataFrame,
    use_trend: bool,
    use_macd: bool,
    use_rsi: bool,
    initial_cash: float,
    fee_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    result = df.copy()
    result["signal"] = build_combined_signal(result, use_trend, use_macd, use_rsi)
    result["position"] = result["signal"].shift(1).fillna(False).astype(int)
    result["daily_return"] = result["close"].pct_change().fillna(0)

    trades = []
    previous_position = 0
    for _, row in result.iterrows():
        position = int(row["position"])
        if position != previous_position:
            trades.append({"date": row["date"], "action": "Buy" if position == 1 else "Sell", "price": row["close"]})
        previous_position = position

    trade_cost = result["position"].diff().abs().fillna(result["position"]) * fee_rate
    result["strategy_return"] = result["position"] * result["daily_return"] - trade_cost
    result["strategy_equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    total_return = result["strategy_equity"].iloc[-1] / initial_cash - 1
    buy_hold_return = result["buy_hold_equity"].iloc[-1] / initial_cash - 1
    days = max((result["date"].iloc[-1] - result["date"].iloc[0]).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    max_drawdown = (result["strategy_equity"] / result["strategy_equity"].cummax() - 1).min()
    volatility = result["strategy_return"].std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility else 0

    metrics = BacktestMetrics(
        total_return=float(total_return),
        buy_hold_return=float(buy_hold_return),
        annual_return=float(annual_return),
        max_drawdown=float(max_drawdown),
        sharpe=float(sharpe),
        trades=len(trades),
        final_equity=float(result["strategy_equity"].iloc[-1]),
    )
    return result, pd.DataFrame(trades), metrics


def generate_parameter_candidates(sample_count: int) -> list[dict[str, int]]:
    rng = np.random.default_rng(42)
    candidates = [THEORETICAL_PARAMS.copy()]
    for _ in range(sample_count):
        fast_ema = int(rng.integers(3, 31))
        slow_ema = int(rng.integers(fast_ema + 5, 121))
        rsi_entry_floor = int(rng.integers(20, 66))
        candidates.append(
            {
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "signal_ema": int(rng.integers(3, 31)),
                "rsi_period": int(rng.integers(5, 41)),
                "rsi_entry_floor": rsi_entry_floor,
                "rsi_overheat_exit": int(rng.integers(rsi_entry_floor + 5, 96)),
            }
        )
    return list({tuple(params.items()): params for params in candidates}.values())


def evaluate_parameter_sets(
    stock_df: pd.DataFrame,
    trend_df: pd.DataFrame | None,
    trend_column: str | None,
    use_trend: bool,
    use_macd: bool,
    use_rsi: bool,
    initial_cash: float,
    fee_rate: float,
    sample_count: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], pd.DataFrame]:
    records = []
    for params in generate_parameter_candidates(sample_count):
        indicator_df = add_indicators(
            stock_df,
            params["fast_ema"],
            params["slow_ema"],
            params["signal_ema"],
            params["rsi_period"],
            params["rsi_entry_floor"],
            params["rsi_overheat_exit"],
            trend_df if use_trend else None,
            trend_column if use_trend else None,
        )
        _, _, metrics = run_backtest(indicator_df, use_trend, use_macd, use_rsi, initial_cash, fee_rate)
        records.append(
            {
                **params,
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "max_drawdown": metrics.max_drawdown,
                "sharpe": metrics.sharpe,
                "trades": metrics.trades,
                "final_equity": metrics.final_equity,
            }
        )

    results = pd.DataFrame(records)
    best_row = results.sort_values(["total_return", "sharpe"], ascending=False).iloc[0]
    worst_row = results.sort_values(["total_return", "sharpe"], ascending=True).iloc[0]
    theoretical_row = results[
        (results["fast_ema"] == THEORETICAL_PARAMS["fast_ema"])
        & (results["slow_ema"] == THEORETICAL_PARAMS["slow_ema"])
        & (results["signal_ema"] == THEORETICAL_PARAMS["signal_ema"])
        & (results["rsi_period"] == THEORETICAL_PARAMS["rsi_period"])
        & (results["rsi_entry_floor"] == THEORETICAL_PARAMS["rsi_entry_floor"])
        & (results["rsi_overheat_exit"] == THEORETICAL_PARAMS["rsi_overheat_exit"])
    ].iloc[0]
    keys = list(THEORETICAL_PARAMS.keys())
    best_params = {key: int(best_row[key]) for key in keys}
    worst_params = {key: int(worst_row[key]) for key in keys}
    theoretical_params = {key: int(theoretical_row[key]) for key in keys}
    return best_params, worst_params, theoretical_params, results.sort_values("total_return", ascending=False)


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    st.set_page_config(page_title="Finance Strategy Backtester", page_icon="📈", layout="wide")
    st.title("Finance Strategy Backtester")
    st.caption("選擇股票後使用內建 Google Trends 資料，並從 Yahoo Finance 取得股價進行回測。")

    with st.sidebar:
        st.header("回測設定")
        stock_name = st.selectbox("選擇股票", list(TREND_DATASETS.keys()))
        ticker = TREND_DATASETS[stock_name]["ticker"]
        st.text_input("Yahoo Finance 股票代碼", value=ticker, disabled=True)
        start_date = st.date_input("開始時間", value=date(2016, 5, 1))
        end_date = st.date_input("結束時間", value=date(2026, 5, 1))
        initial_cash = st.number_input("初始資金", min_value=1000, value=100000, step=1000)
        fee_rate = st.number_input("單次換倉成本", min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f")

        st.divider()
        st.header("技術指標")
        use_trend = st.checkbox("Google Trends", value=True)
        use_macd = st.checkbox("MACD", value=True)
        use_rsi = st.checkbox("RSI", value=True)

        st.divider()
        st.header("自動參數掃描")
        sample_count = st.slider("候選組數", min_value=20, max_value=500, value=120, step=20)

    if not any([use_trend, use_macd, use_rsi]):
        st.warning("請至少勾選一個指標。")
        st.stop()
    if start_date >= end_date:
        st.error("開始時間必須早於結束時間。")
        st.stop()

    try:
        trend_df = load_trends_dataset(stock_name)
        trend_column = TREND_DATASETS[stock_name]["trend_name"]
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        with st.spinner("正在自動掃描參數組合..."):
            best_params, worst_params, theoretical_params, parameter_results = evaluate_parameter_sets(
                stock_df,
                trend_df,
                trend_column,
                use_trend,
                use_macd,
                use_rsi,
                float(initial_cash),
                float(fee_rate),
                int(sample_count),
            )
        indicator_df = add_indicators(
            stock_df,
            best_params["fast_ema"],
            best_params["slow_ema"],
            best_params["signal_ema"],
            best_params["rsi_period"],
            best_params["rsi_entry_floor"],
            best_params["rsi_overheat_exit"],
            trend_df if use_trend else None,
            trend_column if use_trend else None,
        )
        backtest_df, trades_df, metrics = run_backtest(indicator_df, use_trend, use_macd, use_rsi, float(initial_cash), float(fee_rate))
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        st.stop()

    st.info(f"目前使用：{stock_name}（{ticker}），Google Trends 期間 {trend_df['date'].min().date()} 到 {trend_df['date'].max().date()}。")

    st.subheader("自動掃描結果")
    comparison_rows = []
    for label, params in [("最佳", best_params), ("最差", worst_params), ("理論常用", theoretical_params)]:
        matched = parameter_results[
            (parameter_results["fast_ema"] == params["fast_ema"])
            & (parameter_results["slow_ema"] == params["slow_ema"])
            & (parameter_results["signal_ema"] == params["signal_ema"])
            & (parameter_results["rsi_period"] == params["rsi_period"])
            & (parameter_results["rsi_entry_floor"] == params["rsi_entry_floor"])
            & (parameter_results["rsi_overheat_exit"] == params["rsi_overheat_exit"])
        ].iloc[0]
        comparison_rows.append(
            {
                "組別": label,
                "快線 EMA": params["fast_ema"],
                "慢線 EMA": params["slow_ema"],
                "訊號線 EMA": params["signal_ema"],
                "RSI 週期": params["rsi_period"],
                "RSI 進場下限": params["rsi_entry_floor"],
                "RSI 過熱出場": params["rsi_overheat_exit"],
                "策略報酬": matched["total_return"],
                "年化報酬": matched["annual_return"],
                "最大回撤": matched["max_drawdown"],
                "Sharpe": matched["sharpe"],
                "交易次數": int(matched["trades"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format({"策略報酬": "{:.2%}", "年化報酬": "{:.2%}", "最大回撤": "{:.2%}", "Sharpe": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("查看全部參數掃描結果"):
        st.dataframe(
            parameter_results.style.format(
                {
                    "total_return": "{:.2%}",
                    "annual_return": "{:.2%}",
                    "max_drawdown": "{:.2%}",
                    "sharpe": "{:.2f}",
                    "final_equity": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    metric_cols = st.columns(6)
    metric_cols[0].metric("策略報酬", format_pct(metrics.total_return))
    metric_cols[1].metric("買進持有", format_pct(metrics.buy_hold_return))
    metric_cols[2].metric("年化報酬", format_pct(metrics.annual_return))
    metric_cols[3].metric("最大回撤", format_pct(metrics.max_drawdown))
    metric_cols[4].metric("Sharpe", f"{metrics.sharpe:.2f}")
    metric_cols[5].metric("交易次數", metrics.trades)

    st.subheader(f"{stock_name} 價格與交易訊號")
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["close"], name="Close", mode="lines"))
    buys = trades_df[trades_df["action"] == "Buy"] if not trades_df.empty else pd.DataFrame()
    sells = trades_df[trades_df["action"] == "Sell"] if not trades_df.empty else pd.DataFrame()
    if not buys.empty:
        price_fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"], name="Buy", mode="markers", marker={"symbol": "triangle-up", "size": 12, "color": "#168f5a"}))
    if not sells.empty:
        price_fig.add_trace(go.Scatter(x=sells["date"], y=sells["price"], name="Sell", mode="markers", marker={"symbol": "triangle-down", "size": 12, "color": "#c2413d"}))
    price_fig.update_layout(hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("資金曲線")
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["strategy_equity"], name="Strategy", mode="lines"))
    equity_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["buy_hold_equity"], name="Buy & Hold", mode="lines"))
    equity_fig.update_layout(hovermode="x unified")
    st.plotly_chart(equity_fig, use_container_width=True)

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.write("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd"], name="MACD", mode="lines"))
        macd_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["macd_signal"], name="Signal", mode="lines"))
        macd_fig.add_trace(go.Bar(x=backtest_df["date"], y=backtest_df["macd_hist"], name="Hist"))
        macd_fig.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(macd_fig, use_container_width=True)

    with chart_cols[1]:
        st.write("RSI")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["rsi"], name="RSI", mode="lines"))
        rsi_fig.add_hline(y=best_params["rsi_entry_floor"], line_dash="dash", line_color="#777")
        rsi_fig.add_hline(y=best_params["rsi_overheat_exit"], line_dash="dash", line_color="#c2413d")
        rsi_fig.update_layout(height=320, yaxis_range=[0, 100], hovermode="x unified")
        st.plotly_chart(rsi_fig, use_container_width=True)

    with chart_cols[2]:
        st.write("Google Trends")
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["trend_interest"], name="Trend", mode="lines"))
        trend_fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["trend_ma"], name="Trend MA", mode="lines"))
        trend_fig.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(trend_fig, use_container_width=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.subheader("交易紀錄")
        if trades_df.empty:
            st.info("此期間沒有產生交易。")
        else:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    with detail_cols[1]:
        st.subheader("資料預覽")
        preview_columns = ["date", "close", "signal", "position", "macd", "macd_signal", "rsi", "trend_interest"]
        st.dataframe(backtest_df[preview_columns].tail(30), use_container_width=True, hide_index=True)

    st.download_button(
        "下載回測結果 CSV",
        data=backtest_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{ticker}_backtest.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
