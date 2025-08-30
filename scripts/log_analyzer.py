"""
日志分析和监控工具
提供日志性能分析、异常检测、趋势分析等功能
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class LogEntry:
    """日志条目数据结构"""
    timestamp: datetime
    level: str
    logger: str
    module: str
    function: str
    line: int
    message: str
    execution_time: Optional[float] = None
    custom_data: Optional[Dict] = None
    exception: Optional[Dict] = None


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_dir = Path(log_directory)
        self.log_entries = []
        
    def parse_log_files(self, pattern: str = "*.log") -> List[LogEntry]:
        """解析日志文件"""
        self.log_entries = []
        
        for log_file in self.log_dir.glob(pattern):
            if log_file.name.endswith('_error.log'):
                continue  # 跳过错误日志，避免重复
                
            print(f"解析日志文件: {log_file}")
            
            try:
                # 尝试解析多行JSON格式
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    
                    # 分割多个JSON对象
                    json_objects = self._split_json_objects(content)
                    
                    for json_str in json_objects:
                        entry = self._parse_json_entry(json_str)
                        if entry:
                            self.log_entries.append(entry)
                            
            except Exception as e:
                print(f"解析文件 {log_file} 失败: {e}")
                continue
        
        print(f"共解析 {len(self.log_entries)} 条日志记录")
        return self.log_entries
    
    def _split_json_objects(self, content: str) -> List[str]:
        """分割多个JSON对象"""
        json_objects = []
        current_obj = ""
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                escape_next = False
                current_obj += char
                continue
                
            if char == '\\':
                escape_next = True
                current_obj += char
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
            current_obj += char
            
            # 当大括号平衡且不在字符串中时，说明一个JSON对象结束
            if brace_count == 0 and current_obj.strip() and not in_string:
                json_objects.append(current_obj.strip())
                current_obj = ""
                
        return json_objects
    
    def _parse_json_entry(self, json_str: str) -> Optional[LogEntry]:
        """解析JSON格式的日志条目"""
        try:
            data = json.loads(json_str)
            
            # 解析时间戳
            timestamp_str = data.get('timestamp', '')
            try:
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            return LogEntry(
                timestamp=timestamp,
                level=data.get('level', 'UNKNOWN'),
                logger=data.get('logger', 'UNKNOWN'),
                module=data.get('module', 'UNKNOWN'),
                function=data.get('function', 'UNKNOWN'),
                line=data.get('line', 0),
                message=data.get('message', ''),
                execution_time=data.get('custom', {}).get('execution_time'),
                custom_data=data.get('custom', {}),
                exception=data.get('exception')
            )
            
        except Exception as e:
            # 如果JSON解析失败，尝试作为普通文本处理
            return None
    
    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """解析单行日志"""
        try:
            # 尝试解析JSON格式
            if line.startswith('{'):
                data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00')),
                    level=data.get('level', 'UNKNOWN'),
                    logger=data.get('logger', 'UNKNOWN'),
                    module=data.get('module', 'UNKNOWN'),
                    function=data.get('function', 'UNKNOWN'),
                    line=data.get('line', 0),
                    message=data.get('message', ''),
                    execution_time=data.get('custom', {}).get('execution_time'),
                    custom_data=data.get('custom', {}),
                    exception=data.get('exception')
                )
            
            # 尝试解析结构化格式
            elif '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    timestamp_str = parts[0].strip()
                    level = parts[1].strip()
                    module_func = parts[2].strip() if len(parts) > 2 else 'UNKNOWN'
                    message = '|'.join(parts[3:]).strip()
                    
                    # 解析模块和函数
                    if '.' in module_func:
                        module, function = module_func.rsplit('.', 1)
                        if ':' in function:
                            function, line_num = function.split(':')
                            line_num = int(line_num) if line_num.isdigit() else 0
                        else:
                            line_num = 0
                    else:
                        module = module_func
                        function = 'UNKNOWN'
                        line_num = 0
                    
                    # 解析时间戳
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now()
                    
                    # 提取执行时间
                    execution_time = None
                    if 'execution_time' in message:
                        match = re.search(r'execution_time[=:]\s*([\d.]+)', message)
                        if match:
                            execution_time = float(match.group(1))
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level,
                        logger='UNKNOWN',
                        module=module,
                        function=function,
                        line=line_num,
                        message=message,
                        execution_time=execution_time
                    )
            
        except Exception as e:
            print(f"解析日志行失败: {e}")
            return None
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.log_entries:
            return {}
        
        # 执行时间统计
        execution_times = [entry.execution_time for entry in self.log_entries 
                          if entry.execution_time is not None]
        
        # 日志级别分布
        level_counts = Counter(entry.level for entry in self.log_entries)
        
        # 模块活动统计
        module_counts = Counter(entry.module for entry in self.log_entries)
        
        # 异常统计
        exception_counts = sum(1 for entry in self.log_entries if entry.exception)
        
        # 时间范围
        timestamps = [entry.timestamp for entry in self.log_entries]
        time_range = (min(timestamps), max(timestamps)) if timestamps else (None, None)
        
        return {
            'total_entries': len(self.log_entries),
            'time_range': {
                'start': time_range[0].isoformat() if time_range[0] else None,
                'end': time_range[1].isoformat() if time_range[1] else None,
                'duration_hours': (time_range[1] - time_range[0]).total_seconds() / 3600 
                                 if time_range[0] and time_range[1] else None
            },
            'level_distribution': dict(level_counts),
            'module_activity': dict(module_counts.most_common(10)),
            'performance': {
                'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0,
                'slow_operations': len([t for t in execution_times if t > 1.0]) if execution_times else 0
            },
            'error_rate': {
                'total_exceptions': exception_counts,
                'error_percentage': (level_counts.get('ERROR', 0) / len(self.log_entries)) * 100,
                'warning_percentage': (level_counts.get('WARNING', 0) / len(self.log_entries)) * 100
            }
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常模式"""
        anomalies = []
        
        # 检测执行时间异常
        execution_times = [entry.execution_time for entry in self.log_entries 
                          if entry.execution_time is not None]
        
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            std_time = (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
            threshold = avg_time + 3 * std_time
            
            slow_operations = [entry for entry in self.log_entries 
                             if entry.execution_time and entry.execution_time > threshold]
            
            if slow_operations:
                anomalies.append({
                    'type': 'slow_execution',
                    'description': f'发现 {len(slow_operations)} 个异常慢的操作',
                    'threshold': threshold,
                    'operations': [(op.function, op.execution_time) for op in slow_operations[:5]]
                })
        
        # 检测错误激增
        error_entries = [entry for entry in self.log_entries if entry.level == 'ERROR']
        if error_entries:
            # 按小时分组统计错误
            hourly_errors = defaultdict(int)
            for entry in error_entries:
                hour = entry.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_errors[hour] += 1
            
            avg_errors_per_hour = sum(hourly_errors.values()) / len(hourly_errors)
            high_error_hours = {hour: count for hour, count in hourly_errors.items() 
                               if count > avg_errors_per_hour * 2}
            
            if high_error_hours:
                anomalies.append({
                    'type': 'error_spike',
                    'description': f'发现 {len(high_error_hours)} 个错误激增时段',
                    'avg_errors_per_hour': avg_errors_per_hour,
                    'spike_hours': list(high_error_hours.items())[:5]
                })
        
        return anomalies
    
    def generate_report(self, output_file: str = "log_analysis_report.html") -> str:
        """生成分析报告"""
        metrics = self.get_performance_metrics()
        anomalies = self.detect_anomalies()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>日志分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .anomaly {{ background: #ffe6e6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .good {{ background: #e6ffe6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>日志分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>基础指标</h2>
            <div class="metric">
                <h3>总体统计</h3>
                <p>总日志条数: {metrics.get('total_entries', 0)}</p>
                <p>时间范围: {metrics.get('time_range', {}).get('start', 'N/A')} 至 {metrics.get('time_range', {}).get('end', 'N/A')}</p>
                <p>持续时长: {metrics.get('time_range', {}).get('duration_hours', 0):.2f} 小时</p>
            </div>
            
            <div class="metric">
                <h3>日志级别分布</h3>
                <table>
                    <tr><th>级别</th><th>数量</th><th>百分比</th></tr>
        """
        
        total_entries = metrics.get('total_entries', 1)
        for level, count in metrics.get('level_distribution', {}).items():
            percentage = (count / total_entries) * 100
            html_content += f"<tr><td>{level}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="metric">
                <h3>性能指标</h3>
        """
        
        perf = metrics.get('performance', {})
        html_content += f"""
                <p>平均执行时间: {perf.get('avg_execution_time', 0):.3f}s</p>
                <p>最大执行时间: {perf.get('max_execution_time', 0):.3f}s</p>
                <p>慢操作数量: {perf.get('slow_operations', 0)}</p>
            </div>
            
            <h2>异常检测</h2>
        """
        
        if anomalies:
            for anomaly in anomalies:
                html_content += f"""
                <div class="anomaly">
                    <h3>⚠️ {anomaly['type']}</h3>
                    <p>{anomaly['description']}</p>
                </div>
                """
        else:
            html_content += '<div class="good">✅ 未检测到异常模式</div>'
        
        html_content += """
            <h2>模块活动排行</h2>
            <table>
                <tr><th>模块</th><th>日志数量</th></tr>
        """
        
        for module, count in metrics.get('module_activity', {}).items():
            html_content += f"<tr><td>{module}</td><td>{count}</td></tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def create_visualizations(self, output_dir: str = "log_charts"):
        """创建可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.log_entries:
            print("没有日志数据可用于可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 日志级别分布饼图
        level_counts = Counter(entry.level for entry in self.log_entries)
        plt.figure(figsize=(10, 6))
        plt.pie(level_counts.values(), labels=level_counts.keys(), autopct='%1.1f%%')
        plt.title('日志级别分布')
        plt.savefig(output_path / 'log_level_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 时间序列图
        if len(self.log_entries) > 1:
            df = pd.DataFrame([
                {'timestamp': entry.timestamp, 'level': entry.level} 
                for entry in self.log_entries
            ])
            df['hour'] = df['timestamp'].dt.floor('h')
            hourly_counts = df.groupby(['hour', 'level']).size().unstack(fill_value=0)
            
            plt.figure(figsize=(15, 8))
            hourly_counts.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('每小时日志数量趋势')
            plt.xlabel('时间')
            plt.ylabel('日志数量')
            plt.xticks(rotation=45)
            plt.legend(title='日志级别')
            plt.tight_layout()
            plt.savefig(output_path / 'hourly_log_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 执行时间分布直方图
        execution_times = [entry.execution_time for entry in self.log_entries 
                          if entry.execution_time is not None and entry.execution_time < 10]
        
        if execution_times:
            plt.figure(figsize=(10, 6))
            plt.hist(execution_times, bins=50, alpha=0.7, color='skyblue')
            plt.title('执行时间分布')
            plt.xlabel('执行时间 (秒)')
            plt.ylabel('频次')
            plt.axvline(sum(execution_times)/len(execution_times), color='red', 
                       linestyle='--', label=f'平均值: {sum(execution_times)/len(execution_times):.3f}s')
            plt.legend()
            plt.savefig(output_path / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"图表已保存到 {output_path}")


def analyze_trading_logs(log_directory: str = "logs") -> Dict[str, Any]:
    """分析交易日志的便捷函数"""
    analyzer = LogAnalyzer(log_directory)
    analyzer.parse_log_files()
    
    metrics = analyzer.get_performance_metrics()
    anomalies = analyzer.detect_anomalies()
    
    # 生成报告
    report_file = analyzer.generate_report()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    return {
        'metrics': metrics,
        'anomalies': anomalies,
        'report_file': report_file,
        'total_entries': len(analyzer.log_entries)
    }


if __name__ == "__main__":
    print("=== 日志分析工具演示 ===")
    
    # 分析日志
    results = analyze_trading_logs()
    
    print(f"\n分析完成！")
    print(f"总日志条数: {results['total_entries']}")
    print(f"检测到 {len(results['anomalies'])} 个异常模式")
    print(f"报告已生成: {results['report_file']}")
    
    # 显示关键指标
    metrics = results['metrics']
    if metrics:
        print(f"\n关键指标:")
        print(f"  错误率: {metrics.get('error_rate', {}).get('error_percentage', 0):.2f}%")
        print(f"  平均执行时间: {metrics.get('performance', {}).get('avg_execution_time', 0):.3f}s")
        print(f"  慢操作数: {metrics.get('performance', {}).get('slow_operations', 0)}")
    
    # 显示异常
    if results['anomalies']:
        print(f"\n检测到的异常:")
        for anomaly in results['anomalies']:
            print(f"  - {anomaly['type']}: {anomaly['description']}")
    else:
        print("\n✅ 未检测到异常模式")
