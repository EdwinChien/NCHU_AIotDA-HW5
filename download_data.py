from datasets import load_dataset
import pandas as pd

dataset = load_dataset("yaful/MAGE", split="train")

# 只保留必要欄位
df = dataset.to_pandas()[["text", "label"]]

# label: 0 = human, 1 = AI
df = df.dropna()
df = df.sample(50000, random_state=42)  # demo 用，避免太大

df.to_csv("data/train.csv", index=False)
print("Saved data/train.csv")
