# Sarima Branch

## 1. open_prices.csv – Exploratory Data Analysis (EDA)

This section describes the exploratory data analysis (EDA) performed on the **`open_prices.csv`** dataset, along with the limitations we encountered and the motivation for sourcing a new dataset.

---

### 1. Data Cleaning and Filtering

We began by preprocessing the dataset to ensure data quality and relevance for analysis:

* Removed unnecessary or irrelevant columns to reduce noise.
* Added required columns where needed for downstream analysis.
* Checked for missing values, particularly in the **price-related fields**.
* Rows containing **null values in price columns** were removed, as they are not usable for pricing or discount analysis.

This step ensured a clean and consistent dataset suitable for visualization and modeling.

---

### 2. Data Visualization

To better understand the structure and trends in the data, several visualizations were created using **Matplotlib**:

* **Scatter Plot**:

  * Compared *actual price* vs *discounted price* to understand discount behavior.

* **Bar Plots**:

  * Used to visualize and compare pricing trends across **different years**.

* **Line Plots**:

  * Showed price variations over time for multiple years, helping identify trends and seasonal patterns.

These visualizations provided key insights into pricing distributions and discount dynamics.

---

### 3. Price-Based Data Segmentation

To better analyze products across different price ranges, we created two separate DataFrames:

* **Low-to-mid price range**: Products priced between **0 and 1,000**.
* **High price range**: Products priced between **1,000 and 10⁶**.

This separation allowed more focused analysis, as pricing behavior varies significantly across these ranges.

---

### 4. Category Label Analysis and Limitations

While analyzing the **category labels**, we encountered a major limitation:

* A large portion of the dataset had **null values** in the category label column.
* This made it difficult to identify and isolate relevant product categories.

Initially, category labels seemed non-essential. However, they became critical once we decided to build a model **specifically for electronic products**.

Upon further inspection, we discovered that the available labeled data primarily belonged to the **fruits category**, making the dataset unsuitable for our intended use case.

---

### 5. Need for a New Dataset

Due to the limitations identified above, we decided to look for a new dataset with the following requirements:

* Inclusion of **electronics category products**.
* Ability to **filter out non-electronics categories**.
* Availability of a **date/time column**, which is essential for **time series analysis**.
* Presence of **price-related features**, including:

  * Actual price
  * Discounted price or discount percentage
* (Optional but beneficial) **Product name**, which could serve as an additional informative feature.

A dataset meeting these criteria is necessary to build a robust **time series forecasting model** focused on electronic products.

---

This EDA phase was crucial in understanding both the strengths and limitations of the `open_prices.csv` dataset and helped guide the next steps of data collection and model development.

## 2. smart_discount_analyzer_dataset_50k.csv and ElectronicsProductsPricingData.md
we will use the both the datafram to create the model , and then check which will gives us the good data 

### 2.1 df_electrnoincs - for the ElectronicsProductsPricingData.cv
  - In thie dataset ,we have found that "id" column has many duplicated values , but with different values of price amountMax and amountMin , so we have diferenced the value and made one column of price difference, from where we have found that when the price was on sale , still sometimes the price amountMax and Price AmountMin is same.
  - we ahve use the heatmap to found the correlation between the price maxAmount and price minAmount
  - In the electronics producst dataset , there were some noises due to the year , that is the year 2014 and 2015 contains very less data as compared with the other others 

### 2.2 smart_discount_analyzer_dataset_50k.csv
 -- here , we do not have remove any columns , and we have change the date set to the datetime format , furthermore , we have checked if there any year data that we do not need , but all the checking were , and there was no removal of the data.
 - from the seaborn lineplot , we have seen that the 2023 contains from jan to june, where the 2024 data contains the data from the june to dec
 - we have use the pandas resample to convert the daily data into weekly and montly by using the mean.
 - we also use the index for months and year in the reviewdate columns , furthermore , we have use the pivot table(rehsaping and summarzing the data in the dataframe), wher we have sent the month index to groupby , and then values(discount) to summarize the numerical value , and agg value which is (aggfunc). we have then use the pivottable to plot 
 - 
