import threading
import time
import json
import websocket
from typing import Optional
import logging

# 设置日志
logger = logging.getLogger('WebSocketDataHandler')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class WebSocketDataHandler:
    def __init__(self, api_key: str, symbol: str, max_retries: int, logger):
        self.api_key = api_key
        self.symbol = symbol
        self.max_retries = max_retries
        self.logger = logger
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.latest_price = None
        self.price_lock = threading.Lock()

    def initialize(self):
        self.logger.info(f"Initializing WebSocket for {self.symbol}")
        # 可选：添加API key或symbol的合法性检查
        if not self.api_key or not self.symbol:
            raise ValueError("API key and symbol must be provided!")

    def on_open(self, ws):
        self.logger.info("WebSocket opened")
        time.sleep(1)  # 确保连接稳定
        if ws.sock and ws.sock.connected:
            self.logger.info("WebSocket fully connected, subscribing to data...")
            subscribe_msg = {
                "action": "subscribe",
                "params": {"symbols": self.symbol}
            }
            ws.send(json.dumps(subscribe_msg))
        else:
            self.logger.error("WebSocket not fully connected on open")

    def on_message(self, ws, message):
        self.logger.info(f"Received message: {message}")
        try:
            data = json.loads(message)
            if 'event' in data and data['event'] == 'price':
                with self.price_lock:
                    self.latest_price = data['price']
                self.logger.info(f"Received price update: {self.latest_price}")
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode message: {message}")
        except KeyError:
            self.logger.error(f"Unexpected message format: {message}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.running = False

    def start(self):
        if not self.running:
            self.initialize()  # 调用初始化方法
            self.running = True
            for attempt in range(self.max_retries):
                try:
                    ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
                    self.logger.info(f"Connecting to: {ws_url}")
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close
                    )
                    self.ws.run_forever()
                    break
                except Exception as e:
                    self.logger.error(f"WebSocket failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        self.logger.critical("Max retries reached, WebSocket failed to start")
                        self.running = False
            self.logger.info(f"WebSocket started for {self.symbol}")

    def stop(self):
        if self.running:
            self.running = False
            if self.ws:
                self.ws.close()
            self.logger.info("WebSocket stopped")

    def get_latest_price(self):
        with self.price_lock:
            return self.latest_price

# 使用示例
if __name__ == "__main__":
    api_key = "0b0751b0b9074152b55ca2c2a502eca8"  # 替换为你的真实 API key
    symbol = "AAPL"
    max_retries = 5

    handler = WebSocketDataHandler(api_key, symbol, max_retries, logger)
    handler.start()

    time.sleep(60)  # 运行 60 秒
    handler.stop()