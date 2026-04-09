import pandas as pd

df = pd.read_csv("attendance.csv")
df = df.dropna(subset=["Name"])  # Remove rows with empty names

# Extract date from Time
df["Date"] = pd.to_datetime(df["Time"]).dt.date

# Keep only the first entry per person per day
df_cleaned = df.drop_duplicates(subset=["Name", "Date"], keep="first")

# Drop the helper Date column
df_cleaned = df_cleaned.drop(columns=["Date"])

df_cleaned.to_csv("attendance.csv", index=False)