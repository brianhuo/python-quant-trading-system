#!/usr/bin/env python3
"""
IBKR美股交易专用回测增强模块
针对Interactive Brokers的费率结构和交易规则进行优化
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class IBKRCostModel:
    """IBKR真实费率模型"""
    
    def __init__(self):
        # IBKR Lite佣金结构 (2024年费率)
        self.commission_per_share = 0.005  # $0.005/股
        self.min_commission = 1.0          # 最低$1/笔
        self.max_commission_rate = 0.01    # 最高1%
        
        # SEC费用和交易费
        self.sec_fee_rate = 0.0000278      # SEC费用 $0.0278/$1000
        self.trading_activity_fee = 0.000145  # TAF费用
        
        # 滑点模型 (基于真实IBKR执行质量)
        self.slippage_model = {
            'market_order': {'mean': 0.0008, 'std': 0.0003},
            'limit_order': {'mean': 0.0002, 'std': 0.0001}
        }
    
    def calculate_commission(self, shares, price):
        """计算IBKR真实佣金"""
        trade_value = shares * price
        
        # 基础佣金
        base_commission = shares * self.commission_per_share
        
        # 应用最小和最大限制
        commission = max(base_commission, self.min_commission)
        commission = min(commission, trade_value * self.max_commission_rate)
        
        # 添加监管费用
        sec_fee = trade_value * self.sec_fee_rate
        taf_fee = min(shares * self.trading_activity_fee, 8.30)  # TAF上限$8.30
        
        total_cost = commission + sec_fee + taf_fee
        return total_cost
    
    def calculate_slippage(self, price, shares, order_type='market'):
        """计算基于订单类型的滑点"""
        model = self.slippage_model.get(order_type, self.slippage_model['market_order'])
        
        # 考虑订单大小对滑点的影响
        size_impact = min(shares / 10000, 0.001)  # 大单影响
        base_slippage = np.random.normal(model['mean'], model['std'])
        
        return abs(base_slippage + size_impact)

class IBKRBacktesterEnhancement:
    """IBKR专用回测增强功能"""
    
    def __init__(self, account_size=25000):
        self.account_size = account_size
        self.cost_model = IBKRCostModel()
        self.is_pdt_account = account_size >= 25000  # PDT规则
        self.daily_trades = 0
        self.current_date = None
        
        # 月度统计
        self.monthly_returns = {}
        self.monthly_trades = {}
    
    def check_pdt_compliance(self, current_date, proposed_trade):
        """检查PDT(Pattern Day Trading)合规性"""
        if self.is_pdt_account:
            return True  # PDT账户无限制
        
        # 非PDT账户：3个交易日内最多3笔日内交易
        if self.current_date != current_date:
            self.daily_trades = 0
            self.current_date = current_date
        
        if proposed_trade == 'day_trade' and self.daily_trades >= 3:
            return False
        
        return True
    
    def calculate_margin_requirement(self, positions, prices):
        """计算IBKR保证金要求"""
        total_long_value = 0
        total_short_value = 0
        
        for pos in positions:
            if pos['type'] == 'LONG':
                total_long_value += pos['shares'] * prices[pos['symbol']]
            else:
                total_short_value += pos['shares'] * prices[pos['symbol']]
        
        # IBKR股票保证金要求 (通常25%)
        margin_requirement = (total_long_value + total_short_value) * 0.25
        return margin_requirement
    
    def calculate_monthly_metrics(self, results_df):
        """计算月度统计指标"""
        # 按月分组计算收益
        monthly_data = results_df.resample('M', on='date').agg({
            'portfolio_value': ['first', 'last'],
            'trades_count': 'sum'
        }).fillna(0)
        
        monthly_returns = {}
        monthly_win_rate = {}
        
        for month in monthly_data.index:
            start_value = monthly_data.loc[month, ('portfolio_value', 'first')]
            end_value = monthly_data.loc[month, ('portfolio_value', 'last')]
            
            if start_value > 0:
                monthly_return = (end_value - start_value) / start_value
                monthly_returns[month.strftime('%Y-%m')] = monthly_return
                
                # 月度胜率 (该月是否盈利)
                monthly_win_rate[month.strftime('%Y-%m')] = 1 if monthly_return > 0 else 0
        
        # 计算总体月度胜率
        total_months = len(monthly_win_rate)
        winning_months = sum(monthly_win_rate.values())
        overall_monthly_win_rate = winning_months / total_months if total_months > 0 else 0
        
        return {
            'monthly_returns': monthly_returns,
            'monthly_win_rate': overall_monthly_win_rate,
            'winning_months': winning_months,
            'total_months': total_months
        }
    
    def assess_target_feasibility(self, backtest_results):
        """评估目标可行性"""
        annual_return = backtest_results.get('annual_return', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        monthly_win_rate = backtest_results.get('monthly_win_rate', 0)
        
        # 目标值
        target_annual_return = 0.30
        target_max_drawdown = 0.10
        target_monthly_win_rate = 0.65
        
        assessment = {
            'annual_return': {
                'achieved': annual_return,
                'target': target_annual_return,
                'gap': annual_return - target_annual_return,
                'feasible': annual_return >= target_annual_return * 0.8  # 80%达成视为可行
            },
            'max_drawdown': {
                'achieved': max_drawdown,
                'target': target_max_drawdown,
                'gap': target_max_drawdown - max_drawdown,
                'feasible': max_drawdown <= target_max_drawdown * 1.2  # 20%容差
            },
            'monthly_win_rate': {
                'achieved': monthly_win_rate,
                'target': target_monthly_win_rate,
                'gap': monthly_win_rate - target_monthly_win_rate,
                'feasible': monthly_win_rate >= target_monthly_win_rate * 0.9  # 90%达成
            }
        }
        
        # 综合可行性评估
        feasible_count = sum(1 for metric in assessment.values() if metric['feasible'])
        overall_feasible = feasible_count >= 2  # 至少2/3指标达成
        
        assessment['overall'] = {
            'feasible': overall_feasible,
            'score': feasible_count / 3,
            'recommendation': self._get_recommendation(assessment)
        }
        
        return assessment
    
    def _get_recommendation(self, assessment):
        """基于评估结果给出建议"""
        feasible_metrics = [k for k, v in assessment.items() 
                          if k != 'overall' and v['feasible']]
        
        if len(feasible_metrics) == 3:
            return "目标完全可行，建议实盘测试"
        elif len(feasible_metrics) == 2:
            unfeasible = [k for k, v in assessment.items() 
                         if k != 'overall' and not v['feasible']][0]
            return f"基本可行，需优化{unfeasible.replace('_', ' ')}指标"
        elif len(feasible_metrics) == 1:
            return "目标过于激进，建议调整预期或优化策略"
        else:
            return "目标不现实，需要重新制定交易计划"

def create_ibkr_enhanced_report(standard_report, trades_df=None):
    """创建IBKR增强回测报告"""
    enhancer = IBKRBacktesterEnhancement()
    
    # 模拟月度数据 (实际使用时从真实数据计算)
    if trades_df is not None:
        monthly_metrics = enhancer.calculate_monthly_metrics(trades_df)
    else:
        # 使用模拟数据演示
        monthly_metrics = {
            'monthly_win_rate': 0.58,  # 58%月度胜率
            'winning_months': 7,
            'total_months': 12
        }
    
    # 增强报告
    enhanced_report = standard_report.copy()
    enhanced_report.update({
        'monthly_win_rate': monthly_metrics['monthly_win_rate'],
        'winning_months': monthly_metrics.get('winning_months', 0),
        'total_months': monthly_metrics.get('total_months', 0),
        'ibkr_specific': {
            'account_type': 'PDT' if enhancer.is_pdt_account else 'Non-PDT',
            'estimated_annual_costs': enhancer.cost_model.min_commission * 252,  # 估算年度最低成本
            'margin_rate': 'Prime + 1.5%',  # IBKR保证金利率
            'pdt_compliant': enhancer.is_pdt_account
        }
    })
    
    # 目标可行性评估
    feasibility = enhancer.assess_target_feasibility(enhanced_report)
    enhanced_report['target_assessment'] = feasibility
    
    return enhanced_report

# 测试函数
def test_ibkr_enhancement():
    """测试IBKR增强功能"""
    print("=" * 60)
    print("IBKR美股交易回测增强测试")
    print("=" * 60)
    
    # 模拟标准回测结果
    mock_standard_report = {
        'annual_return': 0.28,    # 28%年化收益
        'max_drawdown': 0.12,     # 12%最大回撤
        'win_rate': 0.72,         # 72%整体胜率
        'sharpe_ratio': 1.65
    }
    
    # 生成增强报告
    enhanced_report = create_ibkr_enhanced_report(mock_standard_report)
    
    print("1. 标准指标:")
    print(f"   年化收益率: {enhanced_report['annual_return']:.2%}")
    print(f"   最大回撤: {enhanced_report['max_drawdown']:.2%}")
    print(f"   整体胜率: {enhanced_report['win_rate']:.2%}")
    print(f"   夏普比率: {enhanced_report['sharpe_ratio']:.2f}")
    
    print("\n2. IBKR专用指标:")
    print(f"   月度胜率: {enhanced_report['monthly_win_rate']:.2%}")
    print(f"   盈利月份: {enhanced_report['winning_months']}/{enhanced_report['total_months']}")
    
    print("\n3. 目标可行性评估:")
    assessment = enhanced_report['target_assessment']
    for metric, data in assessment.items():
        if metric != 'overall':
            status = "✅" if data['feasible'] else "❌"
            print(f"   {metric.replace('_', ' ').title()}: {status} ({data['achieved']:.2%} vs {data['target']:.2%})")
    
    print(f"\n4. 综合评估:")
    overall = assessment['overall']
    print(f"   可行性评分: {overall['score']:.1%}")
    print(f"   建议: {overall['recommendation']}")
    
    return enhanced_report

if __name__ == "__main__":
    result = test_ibkr_enhancement()
    print(f"\n完整报告:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
