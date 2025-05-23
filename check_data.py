import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.filter import NameDFilter


def check_data_availability():
    # Initialize qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/etf_data", region=REG_CN)

    # Define the symbol to check
    symbol = "SH512890"

    # Get the list of available instruments
    instruments = D.list_instruments(
        instruments={
            "market": "all",
            "filter_pipe": [
                {
                    "filter_type": "NameDFilter",
                    "name_rule_re": "SH512890",
                    "filter_start_time": None,
                    "filter_end_time": None,
                }
            ],
        }
    )

    print(f"\nAvailable instruments: {len(instruments)}")

    # Check if our symbol exists
    if symbol in instruments:
        print(f"\nSymbol {symbol} is available in the dataset")
    else:
        print(f"\nSymbol {symbol} is NOT available in the dataset")
        return

    # Get the date range for the symbol
    start_date = D.calendar(start_time="2010-01-01", end_time="2024-12-31")[0]
    end_date = D.calendar(start_time="2010-01-01", end_time="2024-12-31")[-1]
    print(f"\nData date range:")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    # Get all available features
    print("\nAvailable features:")
    features = D.list_instruments(
        instruments={
            "market": "all",
            "filter_pipe": [
                {
                    "filter_type": "NameDFilter",
                    "name_rule_re": "sh512890",
                    "filter_start_time": None,
                    "filter_end_time": None,
                }
            ],
        }
    )
    print(features)

    # Try to get some sample data with all features
    print("\nTrying to fetch sample data with all features...")
    df = D.features(
        [symbol],
        ["$close", "$open", "$high", "$low", "$volume", "$factor"],
        start_date,
        end_date,
    )
    if not df.empty:
        print("\nSample data preview:")
        print(df.head())
        print("\nData shape:", df.shape)
        print("\nColumns:", df.columns)
    else:
        print("No data available for the symbol")


if __name__ == "__main__":
    check_data_availability()
