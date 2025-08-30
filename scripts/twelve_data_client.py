import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from dateutil import tz
import os
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwelveDataClient:
    def __init__(self, api_key=None, max_retries=3):
        load_dotenv()
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 Twelve Data API Key，请设置环境变量 TWELVE_DATA_API_KEY 或传入 api_key")
        self.max_retries = max_retries
        self.base_url = "https://api.twelvedata.com/time_series"
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_file(self, symbol, interval, start_date, end_date):
        return os.path.join(self.cache_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")

    def _estimate_points_per_day(self, interval):
        return {"1min": 1440, "1h": 24, "1day": 1}.get(interval, 1)

    def _safe_tz_localize(self, index, timezone):
        if isinstance(index, pd.DatetimeIndex) and index.tz is None:
            return index.tz_localize(timezone)
        return index

    def get_historical_data(self, symbol, interval, start_date, end_date, timezone="America/New_York", outputsize=None, retries=3, timeout=10):
        cache_file = self._get_cache_file(symbol, interval, start_date, end_date)
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col="datetime", parse_dates=True)
                df.index = self._safe_tz_localize(df.index, timezone)
                # 将 start_date 和 end_date 转换为带时区的对象
                start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz.gettz(timezone))
                end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz.gettz(timezone))
                if df.index.min() <= start and df.index.max() >= end:
                    logger.info(f"从缓存加载数据: {cache_file}")
                    return df
            except Exception as e:
                logger.warning(f"缓存文件读取失败: {e}")

        all_data = []
        try:
            # 为 start 和 end 添加时区信息
            start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz.gettz(timezone))
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz.gettz(timezone))
        except ValueError as e:
            logger.error(f"无效的日期格式: {e}")
            return pd.DataFrame()

        points_per_day = self._estimate_points_per_day(interval)
        total_days = (end - start).days + 1
        total_points = total_days * points_per_day
        max_output_size = outputsize if outputsize is not None else 5000

        current_start = start
        while current_start <= end:
            if total_points > max_output_size:
                batch_days = max_output_size // points_per_day
                current_end = min(current_start + timedelta(days=batch_days - 1), end)
            else:
                current_end = end

            df_batch = self._get_data_batch(symbol, interval, current_start.strftime("%Y-%m-%d"), 
                                          current_end.strftime("%Y-%m-%d"), timezone, outputsize, retries, timeout)
            if not df_batch.empty:
                all_data.append(df_batch)
            current_start = current_end + timedelta(days=1)

        if all_data:
            try:
                full_df = pd.concat(all_data)
                full_df = self._clean_and_validate_data(full_df)
                full_df.to_csv(cache_file)
                logger.info(f"数据已缓存到: {cache_file}")
                return full_df
            except Exception as e:
                logger.error(f"数据合并失败: {e}")

        logger.warning("未检索到有效数据")
        return pd.DataFrame()

    def _get_data_batch(self, symbol, interval, start_date, end_date, timezone, outputsize, retries, timeout):
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "outputsize": outputsize if outputsize is not None else 5000,
            "timezone": timezone
        }
        for attempt in range(retries):
            try:
                time.sleep(0.1)  # 频率控制
                response = requests.get(self.base_url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if data.get("status") != "ok":
                    logger.error(f"API错误: {data.get('code', 'Unknown')} - {data.get('message')}")
                    return pd.DataFrame()
                if not data.get("values"):
                    logger.info(f"{symbol} 在 {start_date} 至 {end_date} 无数据")
                    return pd.DataFrame()
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(timezone)
                df.set_index("datetime", inplace=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                return df
            except requests.exceptions.RequestException as e:
                logger.warning(f"第 {attempt+1}/{retries} 次尝试失败: {e}")
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except Exception as e:
                logger.error(f"意外错误: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _clean_and_validate_data(self, df):
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        return df

    def get_latest_kline(self, symbol, interval, start_date, end_date, timezone="America/New_York", retries=3, timeout=10):
        df = self.get_historical_data(symbol, interval, start_date, end_date, timezone, retries=timeout)
        if not df.empty:
            return df.iloc[-1]
        return None

if __name__ == "__main__":
    client = TwelveDataClient()
    print("测试 get_historical_data:")
    df = client.get_historical_data(
        symbol="AAPL",
        interval="1day",
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        timezone="America/New_York",
        outputsize=5000  # 添加 outputsize 参数进行测试
    )
    if not df.empty:
        print(f"历史数据（前5行）:\n{df.head()}")
    else:
        print("无历史数据返回")

    print("\n测试 get_latest_kline:")
    latest = client.get_latest_kline(
        symbol="AAPL",
        interval="1day",
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        timezone="America/New_York",
    )
    if latest is not None:
        print(f"最新数据:\n{latest}")
    else:
        print("无最新数据返回")

    print("\n测试无效符号:")
    invalid_data = client.get_latest_kline(
        symbol="INVALID_SYMBOL",
        interval="1day",
        start_date="2020-01-01",
        end_date="2025-04-17",
    )
    if invalid_data is not None:
        print(f"无效符号数据:\n{invalid_data}")
    else:
        print("无效符号无数据返回")