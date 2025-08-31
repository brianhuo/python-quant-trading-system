#!/usr/bin/env python3
"""
干净演示版本 - 交易数据管道
无警告运行版本
"""

import warnings
import urllib3

# 抑制所有警告
warnings.filterwarnings('ignore')
urllib3.disable_warnings()

from trading_data_pipeline import create_default_pipeline
import time

def main():
    print("🎯 交易数据管道 - 无警告演示")
    print("=" * 50)
    
    try:
        # 创建数据管道
        print("🔧 初始化数据管道...")
        pipeline = create_default_pipeline()
        
        # 单股票演示
        print("\n📈 获取AAPL数据...")
        start_time = time.time()
        data, report = pipeline.get_clean_data("AAPL", "30min", 100)
        processing_time = time.time() - start_time
        
        if report["status"] == "success":
            print(f"✅ 成功: {data.shape[0]} 行数据")
            print(f"⏱️  处理时间: {processing_time:.3f}秒")
            print(f"📊 数据范围: {data.index[0]} 到 {data.index[-1]}")
            
            # 显示数据质量
            health_report = report.get("health_report")
            if health_report:
                summary = health_report.get_summary()
                print(f"🔍 健康状态: {summary['health_status']}")
                print(f"📈 数据完整性: {summary['completeness_rate']:.1%}")
        else:
            print(f"❌ 失败: {report.get('error')}")
            
        # 批量处理演示
        print("\n🚀 批量处理演示...")
        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = {}
        
        for symbol in symbols:
            try:
                data, report = pipeline.get_clean_data(symbol, "30min", 50)
                results[symbol] = {"status": "success", "data": data, "report": report}
            except Exception as e:
                results[symbol] = {"status": "error", "error": str(e)}
        
        print(f"📊 批量结果: {len(results)} 个股票")
        success_count = 0
        total_rows = 0
        
        for symbol, result in results.items():
            if result["status"] == "success":
                rows = result['data'].shape[0]
                total_rows += rows
                success_count += 1
                print(f"   {symbol}: ✅ {rows} 行")
            else:
                print(f"   {symbol}: ❌ {result.get('error', 'Unknown error')}")
                
        # 显示管道统计
        print(f"\n📈 管道统计:")
        print(f"   成功率: {success_count/len(symbols):.1%}")
        print(f"   总数据行数: {total_rows}")
        print(f"   平均每股票: {total_rows/success_count:.0f} 行" if success_count > 0 else "   平均每股票: 0 行")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
    finally:
        # 清理资源
        if 'pipeline' in locals() and hasattr(pipeline, 'data_client'):
            try:
                pipeline.data_client.close()
            except:
                pass
            
    print("\n🎉 演示完成！")
    print("\n💡 使用建议:")
    print("   # 在您的交易策略中:")
    print("   pipeline = create_default_pipeline()")
    print("   clean_data, report = pipeline.get_clean_data('SYMBOL', '30min', 1000)")
    print("   # 数据已清洗完毕，可直接用于分析")

if __name__ == "__main__":
    main()
