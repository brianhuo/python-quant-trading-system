#!/usr/bin/env python3
"""
测试不同环境配置
"""

from enhanced_config_loader import load_config

def test_environments():
    print('=== 测试不同环境配置 ===')
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        try:
            config = load_config(environment=env, validate=False)
            print(f'{env.upper()}:')
            print(f'  资金: ${config["INIT_CAPITAL"]:,}')
            print(f'  风险: {config["RISK_PER_TRADE"]*100}%')
            print(f'  实盘: {config["LIVE_TRADING"]}')
            print(f'  日志: {config["LOG_LEVEL"]}')
            print()
        except Exception as e:
            print(f'{env} 环境加载失败: {e}')
            print()

if __name__ == "__main__":
    test_environments()
