import argparse
import pandas as pd

DATA_PATH = "data/data.csv"
NROWS_SAMPLE = 5000  # дефолт: быстрый сэмпл


def load_df(full: bool) -> pd.DataFrame:
    if full:
        print(f"Читаю весь датасет из {DATA_PATH} (может занять время)...")
        return pd.read_csv(DATA_PATH)
    print(f"Читаю сэмпл {NROWS_SAMPLE} строк из {DATA_PATH}...")
    return pd.read_csv(DATA_PATH, nrows=NROWS_SAMPLE)


def main(full: bool):
    df = load_df(full)
    print("Форма:", df.shape)
    print("Колонки (первые 10):", df.columns[:10].tolist())
    print("\nБаланс таргета:")
    print(df["target"].value_counts(normalize=True))
    print("\nТоп-20 колонок по пропускам:")
    missing = df.isnull().mean().sort_values(ascending=False)
    print(missing.head(20))

    print("\nИнформация о данных:")
    print(df.info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Быстрый EDA датасета")
    parser.add_argument(
        "--full",
        action="store_true",
        help="читать весь файл (без сэмпла); медленнее и больше памяти",
    )
    args = parser.parse_args()
    main(full=args.full)
