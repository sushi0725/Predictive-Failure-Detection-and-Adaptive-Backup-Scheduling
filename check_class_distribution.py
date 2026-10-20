
import sys
sys.path.insert(0, "src")
import dataset
df = dataset.load_labelled_features()
train_df, test_df = dataset.chronological_split(df)
for name, part in [("Training", train_df), ("Testing", test_df)]:
    print(f"\n{name} set")
    print(f"Rows: {len(part):,}")
    print(f"Failures: {(part['fails_within_horizon'] == 1).sum():,}")
    print(f"Healthy: {(part['fails_within_horizon'] == 0).sum():,}")
    print(f"Date range: {part['feature_date'].min()} to {part['feature_date'].max()}")
