# run_backtest.py

from backtest import BacktestEngine
import logging
import time

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('BacktestRunner')

    try:
        # Initialize backtest engine
        engine = BacktestEngine(initial_capital=10000)  # 10,000 USDT

        # Load historical data
        data = engine.load_data(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date='2019-09-08',
            end_date='2022-07-27'
        )

        if data is None:
            logger.error("Failed to load historical data")
            return

        # Run backtest
        results = engine.run_backtest(data)
        if results is None:
            logger.error("Backtest failed")
            return

        # Generate and print report
        report = engine.generate_report()
        print(report)

        # Generate and save plot
        fig = engine.plot_results()
        fig.write_html("backtest_results.html")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")

if __name__ == "__main__":
    main()