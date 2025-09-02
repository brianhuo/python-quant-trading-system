#!/usr/bin/env python3
"""
配置对比工具 - 比较不同环境的配置差异
"""

from enhanced_config_loader import load_config
import json

def compare_configs():
    """比较不同环境的配置"""
    print('=' * 60)
    print('🔍 配置环境对比分析')
    print('=' * 60)
    
    environments = ['development', 'testing', 'production']
    configs = {}
    
    # 加载所有环境配置
    for env in environments:
        try:
            configs[env] = load_config(environment=env, validate=False)
            print(f'✅ {env} 环境配置已加载')
        except Exception as e:
            print(f'❌ {env} 环境配置加载失败: {e}')
            return
    
    print()
    
    # 关键参数对比
    key_params = [
        ('INIT_CAPITAL', '初始资金', lambda x: f'${x:,}'),
        ('RISK_PER_TRADE', '单笔风险', lambda x: f'{x*100:.1f}%'),
        ('LIVE_TRADING', '实盘交易', lambda x: '是' if x else '否'),
        ('LOG_LEVEL', '日志级别', str),
        ('STOPLOSS_MULTIPLIER', '止损倍数', lambda x: f'{x}x'),
        ('TAKEPROFIT_MULTIPLIER', '止盈倍数', lambda x: f'{x}x'),
        ('MAX_DRAWDOWN', '最大回撤', lambda x: f'{x*100:.0f}%'),
        ('MODEL_UPDATE_INTERVAL_DAYS', '模型更新', lambda x: f'{x}天'),
        ('ROLLING_WINDOW', '滚动窗口', str)
    ]
    
    print(f'{"参数":<15} {"开发环境":<15} {"测试环境":<15} {"生产环境":<15}')
    print('-' * 65)
    
    for param, name, formatter in key_params:
        dev_val = formatter(configs['development'][param])
        test_val = formatter(configs['testing'][param])
        prod_val = formatter(configs['production'][param])
        
        print(f'{name:<15} {dev_val:<15} {test_val:<15} {prod_val:<15}')
    
    print()
    print('🎯 环境特点分析:')
    print('  开发环境: 小资金、低风险、详细日志、快速更新')
    print('  测试环境: 中等资金、中等风险、标准日志、平衡参数')
    print('  生产环境: 完整资金、标准风险、精简日志、稳定参数')
    print()
    
    # 风险对比
    print('⚠️  风险等级对比:')
    for env in environments:
        config = configs[env]
        risk_score = (
            config['RISK_PER_TRADE'] * 50 +
            config['MAX_DRAWDOWN'] * 20 +
            (1 if config['LIVE_TRADING'] else 0) * 30
        )
        risk_level = '高' if risk_score > 2 else '中' if risk_score > 1 else '低'
        print(f'  {env:<12}: 风险等级 {risk_level} (分数: {risk_score:.1f})')

def show_config_validation():
    """显示配置验证功能"""
    print('\n' + '=' * 60)
    print('🔒 配置验证功能演示')
    print('=' * 60)
    
    from config_schema import validate_config
    
    # 测试无效配置
    invalid_config = {
        "trading": {
            "ticker": "INVALID_TICKER_TOO_LONG",
            "exchange": "ARCA",
            "currency": "USD", 
            "initial_capital": -1000,  # 负数
        },
        "risk_management": {
            "risk_per_trade": 1.5,  # 超出范围
            "max_trade_percentage": 0.1,
            "stop_loss_multiplier": 1.8,
            "take_profit_multiplier": 2.5,
            "max_drawdown": 0.1,
            "daily_loss_limit": -0.03
        }
    }
    
    print('测试无效配置...')
    errors = validate_config(invalid_config)
    if errors:
        print('❌ 发现配置错误:')
        for error in errors:
            print(f'   • {error}')
    else:
        print('✅ 配置验证通过')

if __name__ == "__main__":
    compare_configs()
    show_config_validation()





