import glob
import os
import pandas as pd

DATA_DIR = "Data"

os.makedirs(DATA_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(DATA_DIR, "*_backtest_results.csv"))

if not csv_files:
    print("No backtest result CSV files found.")
    print("Run pattern.py first so it creates a file like AAPL_backtest_results.csv.")
    raise SystemExit

print("Available result files:")
for index, file in enumerate(csv_files, start=1):
    print(f"{index}. {file}")

choice = input("Choose file number to analyze: ").strip()

try:
    selected_file = csv_files[int(choice) - 1]
except (ValueError, IndexError):
    print("Invalid choice.")
    raise SystemExit

results = pd.read_csv(selected_file)

if results.empty:
    print("The selected CSV file has no trades.")
    raise SystemExit

required_columns = {"entry_time", "result", "return"}
missing_columns = required_columns - set(results.columns)

if missing_columns:
    print("Missing required columns:", ", ".join(missing_columns))
    raise SystemExit

results["entry_time"] = pd.to_datetime(results["entry_time"])
results["entry_clock"] = results["entry_time"].dt.strftime("%H:%M")
results["entry_hour"] = results["entry_time"].dt.strftime("%H:00")
results["is_win"] = results["result"].eq("WIN")

summary = {
    "file": selected_file,
    "trades": len(results),
    "wins": int(results["is_win"].sum()),
    "losses": int((~results["is_win"]).sum()),
    "win_rate": results["is_win"].mean() * 100,
    "total_return": results["return"].sum() * 100,
    "average_return": results["return"].mean() * 100,
}

by_time = (
    results.groupby("entry_clock")
    .agg(
        trades=("result", "count"),
        wins=("is_win", "sum"),
        win_rate=("is_win", "mean"),
        avg_return=("return", "mean"),
        total_return=("return", "sum"),
    )
    .reset_index()
)

by_time["win_rate"] = by_time["win_rate"] * 100
by_time["avg_return"] = by_time["avg_return"] * 100
by_time["total_return"] = by_time["total_return"] * 100
by_time = by_time.sort_values(["win_rate", "total_return", "trades"], ascending=[False, False, False])

by_hour = (
    results.groupby("entry_hour")
    .agg(
        trades=("result", "count"),
        wins=("is_win", "sum"),
        win_rate=("is_win", "mean"),
        avg_return=("return", "mean"),
        total_return=("return", "sum"),
    )
    .reset_index()
)

by_hour["win_rate"] = by_hour["win_rate"] * 100
by_hour["avg_return"] = by_hour["avg_return"] * 100
by_hour["total_return"] = by_hour["total_return"] * 100
by_hour = by_hour.sort_values(["win_rate", "total_return", "trades"], ascending=[False, False, False])

strong_times = by_time[(by_time["trades"] >= 2) & (by_time["win_rate"] >= 55)]
weak_times = by_time[(by_time["trades"] >= 2) & (by_time["win_rate"] < 50)]

print("\nOverall Summary")
print("File:", summary["file"])
print("Trades:", summary["trades"])
print("Wins:", summary["wins"])
print("Losses:", summary["losses"])
print("Win rate:", round(summary["win_rate"], 2), "%")
print("Total return:", round(summary["total_return"], 2), "%")
print("Average return per trade:", round(summary["average_return"], 2), "%")

print("\nBest entry times")
print(by_time.head(10).to_string(index=False))

print("\nWorst entry times")
print(by_time.sort_values(["win_rate", "total_return", "trades"]).head(10).to_string(index=False))

print("\nBest entry hours")
print(by_hour.head(10).to_string(index=False))

print("\nSuggested allowed entry times")
if strong_times.empty:
    print("No strong times found yet. You need more trades or looser filters.")
else:
    allowed_times = strong_times["entry_clock"].tolist()
    print(allowed_times)

print("\nTimes to avoid")
if weak_times.empty:
    print("No weak times found with enough trades.")
else:
    avoid_times = weak_times.sort_values(["win_rate", "total_return"])["entry_clock"].tolist()
    print(avoid_times)

output_file = selected_file.replace("_backtest_results.csv", "_time_analysis.csv")
by_time.to_csv(output_file, index=False)
print("\nSaved time analysis to", output_file)
