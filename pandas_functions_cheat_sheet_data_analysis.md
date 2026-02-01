# 📊 Pandas Functions Cheat Sheet

*A clean, exam‑friendly, and thesis‑ready reference for everyday data analysis with pandas.*

---

## 🧭 How to Use This Document
- **Learn** what each pandas function does
- **Understand** when and why to use it
- **Reuse** the examples directly in Jupyter Notebook
- **Cite** the methodology in reports or theses

---

## 1️⃣ Loading & Saving Data

### `pd.read_csv()` — Load data
Reads a CSV file into a DataFrame.

```python
df = pd.read_csv("data.csv")
```
**Common options**
- `parse_dates=True` → parse date columns
- `index_col="date"` → set index while loading

---

### `df.to_csv()` — Save data
Writes a DataFrame to disk.

```python
df.to_csv("output.csv", index=False)
```

---

## 2️⃣ Inspecting the Dataset

| Function | Purpose |
|--------|--------|
| `df.head()` | View first rows |
| `df.tail()` | View last rows |
| `df.shape` | Rows × columns |
| `df.columns` | Column names |

```python
df.head()
df.info()
```

---

### `df.info()` — Structure overview
Shows:
- column names
- data types
- non‑null counts

---

### `df.describe()` — Statistical summary
Provides:
- mean, std, min, max
- quartiles (25%, 50%, 75%)

```python
df.describe()
```

---

## 3️⃣ Selecting & Filtering Data

### Column selection
```python
df["price"]
df[["price", "discount"]]
```

### Row filtering
```python
df[df["price"] > 100]
```

### Multiple conditions
```python
df[(df["price"] > 100) & (df["discount"] > 0)]
```

---

## 4️⃣ Data Type Conversion

### Convert to datetime
Used for time‑series analysis.

```python
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
```

---

### Convert strings to numbers
Safely converts text to numeric values.

```python
df["price"] = pd.to_numeric(df["price"], errors="coerce")
```

---

### Remove thousand separators
```python
df["price"] = df["price"].str.replace(",", "").astype(float)
```

---

## 5️⃣ Handling Missing Values

| Function | Use |
|-------|----|
| `isna()` | Detect missing |
| `dropna()` | Remove missing |
| `fillna()` | Replace missing |

```python
df.dropna(subset=["price"])
df["price"].fillna(df["price"].mean())
```

---

## 6️⃣ Creating & Modifying Columns

### Feature engineering

```python
df["discount_percent"] = (
    (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100
)
```

### Rename columns
```python
df.rename(columns={"old": "new"}, inplace=True)
```

---

## 7️⃣ Indexing & Sorting

```python
df.set_index("date", inplace=True)
df.reset_index(inplace=True)
df.sort_values("price", ascending=False)
```

---

## 8️⃣ Grouping & Aggregation

### `groupby()` — Core analysis tool

```python
df.groupby("year")["discount"].mean()
```

### Multiple aggregations

```python
df.groupby("year").agg({
    "price": "mean",
    "discount": "max"
})
```

---

## 9️⃣ Time Series Analysis

### Extract date components
```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
```

### Resampling
```python
df.resample("M").mean()
df.resample("W").sum()
```

### Rolling statistics
```python
df["7d_avg"] = df["price"].rolling(7).mean()
```

---

## 🔟 Visualization (Quick Plots)

| Plot | Example |
|----|----|
| Line | `df.plot()` |
| Scatter | `df.plot.scatter(x="price", y="discount")` |
| Bar | `groupby().plot(kind="bar")` |
| Box | `df.boxplot(by="category")` |

---

## 1️⃣1️⃣ Exporting Results

```python
df.to_csv("cleaned_data.csv", index=False)
```

---

## ✅ Best Practices (Exam & Thesis Ready)

- Always start with `df.info()`
- Convert dates early
- Never ignore missing values
- Use `groupby`, not loops
- Log‑scale skewed price data
- Plot only after cleaning

---

## 🧠 Summary

This cheat sheet covers the **complete pandas workflow**:
- loading → cleaning → transforming → analyzing → visualizing

It is suitable for:
- 📚 exams
- 📊 data analysis projects
- 🎓 academic theses

---

**End of document** ✔️

