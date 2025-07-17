from src.data.config import parse_args
from src.data.data_provider import DataProvider


def main():
    args = parse_args()
    provider = DataProvider(
        symbol=args.symbol, interval=args.interval, start=args.start, end=args.end
    )
    series = provider.fetch()
    print(f"Fetched {len(series.candles)} candles for {args.symbol} [{args.interval}]")


if __name__ == "__main__":
    main()
