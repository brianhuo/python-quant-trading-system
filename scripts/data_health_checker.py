import pandas as pd
import numpy as np
from twelve_data_client import TwelveDataClient
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

class DataHealthChecker:
    def __init__(self, logger=None):
        self.logger = logger
        self.api_call_timestamps = []

    def data_health_check(self, df: pd.DataFrame, raise_on_error: bool = True) -> dict:
        """
        Perform health check on the input DataFrame.

        Parameters:
        - df: Input DataFrame to check.
        - raise_on_error: If True, raise ValueError on critical issues; otherwise, record issues in the report.

        Returns:
        - A dictionary containing the health check report, including an 'issues' list for any problems found.
        """
        report = {
            'missing_ratio': df.isnull().mean(),
            'zero_values': (df == 0).mean(),
            'outliers': df.apply(lambda x: np.mean(np.abs(x - x.mean()) > 5 * x.std())),
            'issues': []
        }

        # Check for high missing value ratio
        if report['missing_ratio'].max() > 0.1:
            issue = f"数据缺失严重: 缺失率 {report['missing_ratio'].max():.2%}"
            if raise_on_error:
                raise ValueError(issue)
            else:
                report['issues'].append(issue)

        # Check for high zero value ratio
        if report['zero_values'].max() > 0.05:
            issue = f"存在较多零值数据: 零值率 {report['zero_values'].max():.2%}"
            if self.logger:
                self.logger.warning(issue)
            report['issues'].append(issue)

        # Check for negative volume values
        if 'volume' in df.columns and (df['volume'] < 0).any():
            issue = "存在负的交易量数据"
            if raise_on_error:
                raise ValueError(issue)
            else:
                report['issues'].append(issue)

        # Check time continuity
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff().dt.total_seconds()
            if (time_diff > 3600 * 24 * 1.5).any():  # Allow 1.5 days error
                report['issues'].append("时间序列不连续")

        # Check API call frequency
        if self.api_call_timestamps:
            one_minute_ago = datetime.now() - pd.Timedelta(minutes=1)
            recent_calls = [ts for ts in self.api_call_timestamps if ts > one_minute_ago]
            if len(recent_calls) > 600:
                if self.logger:
                    self.logger.warning("接近API调用限额")

        # Add feature statistics
        report['feature_stats'] = {
            'volatility': df['close'].pct_change().std(),
            'volume_skew': df['volume'].skew() if 'volume' in df.columns else np.nan
        }

        return report

    def check_data(self, df):
        """
        Check if the data is healthy.

        Parameters:
        - df: Input DataFrame to check.

        Returns:
        - True if the data is healthy, False otherwise.
        """
        # Check if data is empty
        if df.empty:
            if self.logger:
                self.logger.error("数据为空")
            return False
        # Check for missing values
        if df.isnull().any().any():
            if self.logger:
                self.logger.error("数据中存在缺失值")
            return False
        return True

if __name__ == "__main__":
    # Example usage with logger (assuming logger_setup is available)
    try:
        from logger_setup import setup_logging
        logger = setup_logging()
    except ImportError:
        logger = None

    # Get API key from environment variable
    api_key = os.getenv('TWELVE_DATA_API_KEY')
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY 环境变量未设置")

    # Initialize TwelveDataClient
    client = TwelveDataClient(api_key=api_key)

    # Get historical data with outputsize parameter
    df = client.get_historical_data(
        symbol="AAPL",
        interval="1day",
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        outputsize=5000,  # Added support for outputsize
        timezone="America/New_York"
    )

    # Create DataHealthChecker instance
    checker = DataHealthChecker(logger=logger)
    checker.api_call_timestamps.append(datetime.now())

    if df.empty:
        print("未能获取到数据，请检查 API 密钥和网络连接。")
    else:
        # Run check_data
        if checker.check_data(df):
            print("数据健康检查通过。")
        else:
            print("数据健康检查未通过，请检查日志。")

        # Run health check without raising errors
        report = checker.data_health_check(df, raise_on_error=False)
        print("Health Check Report for actual data:")
        for key, value in report.items():
            if key == 'issues':
                print(f"{key}:")
                for issue in value:
                    print(f"  - {issue}")
            else:
                print(f"{key}: {value}")

        # Run health check with raising errors
        try:
            report = checker.data_health_check(df, raise_on_error=True)
            print("数据健康检查通过。")
        except ValueError as e:
            print(f"Error during health check: {e}")