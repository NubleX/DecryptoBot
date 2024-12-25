# DecryptoBot

DecryptoBot is a conservative crypto trading bot focusing on new listings and range trading. It uses the Binance API to trade cryptocurrencies based on technical analysis and predefined trading strategies.

## Features

- Monitors new listings on Binance and trades based on predefined criteria.
- Implements range trading strategies.
- Uses technical analysis to generate trading signals.
- Manages risk with stop-loss and take-profit orders.
- Logs trading activities and errors.

## Requirements

- Python 3.7+
- Binance API keys
- A `.env` file with the necessary environment variables

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/nublex/decryptobot.git
    cd decryptobot
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a [.env](http://_vscodecontentref_/0) file in the root directory and add your Binance API keys and other configuration parameters:

    ```env
    BINANCE_API_KEY=your_api_key
    BINANCE_API_SECRET=your_api_secret
    MAX_LISTING_AGE_SECONDS=300
    LISTING_POSITION_SIZE=50
    LISTING_PROFIT_TARGET=3.0
    LISTING_STOP_LOSS=2.0
    MIN_SIGNAL_STRENGTH=0.6
    MIN_RISK_REWARD_RATIO=1.5
    RANGE_TRADING_ENABLED=True
    MAX_POSITION_SIZE=100
    STOP_LOSS_PERCENTAGE=2.0
    TAKE_PROFIT_PERCENTAGE=5.0
    MAX_TRADES_PER_DAY=10
    ```

## Usage

1. Run the bot:

    ```sh
    python decryptobot.py
    ```

2. The bot will start monitoring new listings and trading based on the configured strategies.


## Logging

The bot logs its activities to `decryptobot.log` and the console. You can check the log file for detailed information about the bot's operations and any errors that occur.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
