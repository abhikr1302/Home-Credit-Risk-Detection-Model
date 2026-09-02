from src.features.aggregations import create_all_features


def main():
    print("=" * 70)
    print("HOME CREDIT RISK DETECTION PIPELINE")
    print("=" * 70)

    create_all_features()

    print("\n" + "=" * 70)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
