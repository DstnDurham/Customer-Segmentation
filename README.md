# Customer Segmentation
Find the dataset [HERE](https://www.gigasheet.com/sample-data/customer-segmentation)

## Overview
Customer segmentation is the process of dividing current and prospective customers into meaningful sub-groups based on shared metrics and traits. Businesses can refine targeted marketing measures and strategies to smaller, specific groups supported by a deep understanding of consumer preferences and needs. 

In dividing markets, researchers search for common characteristics such as demographics, geographical location, and behavior trends. The segments an organization uses vary depending on its size and industry. 

#### **Demographics:**
Demographic segmentation involves dividing consumers based on factors such as age, race, religion, gender, family size, ethnicity, income and education level. For example, if a product is popular among students, demographic segmentation can be used to create distinct marketing messages that differ from those meant to appeal to parents.
#### **Geographics:**
Geographic segmentation includes grouping customers by region, including what county, state, city and town the person lives in. This approach can offer insight into preferred language, cultural practices, and local climate. For instance, an automobile manufacturer might focus on snow tires and All Wheel Drive systems in snowy locations, while seat ventilation and air conditioning may be featured in hotter climates. 
#### **Behavior:**
Behavioral segmentation is based on customer behaviors and interactions with a brand. This considers spending and media habits, hobbies, product use and desired benefits. This could include purchase history, loyalty and rewards programs, and customer website and social media interactions. 
#### **Needs-Based:**
Needs-based segmentation focuses on common consumer needs. This may include requirements for a product or service, delivery methods, or how personal dersires are met. Needs-based segmentations offers more useful products, builds trust with customers, and develops markets for new products and technologies.


#### **Business-to-Business (B2B) and Business-to-Consumer (B2C)**

Though both processess involve taking the needs, behaviors, and characteristics of specific customer groups into consideration, the customer segmentation process may work differently when marketing and selling to consumers, as opposed to other businesses.

- For behavioral segmentation, B2C marketing might consider consumer browsing history, spending habits, and brand interaction. A B2B marketer might study how business contacts interact with one another througout various organizations and across emails.
  
-  B2C marketers might segment customers based on demographic details like income, family or relationship status, and age group. B2B marketers might segment customers based on industry, company size, revenue, and the roles and teams within.
  
## Objective

This project aims to analyze a dataset of 2000 customers covering demographics, income, spending score, profession, work experience, and family size in order to perform customer segmentation. In order to generate actionable insights that support targeted marketing strategies, improve customer engagement, and demonstrate the ability to apply data-driven decision-making techniques in a real-world context.

### Exploratory Analysis
```
df = pd.read_csv('Cust_Seg.csv')
df.head()
```
<img width="810" height="142" alt="image" src="https://github.com/user-attachments/assets/3381ae26-7bbd-4292-8918-7acff6714328" />

Attribute Information:
This data contains 2,000 potential customer records with 8 variables.
- **Customer ID**: The unique customer ID.
- **Gender**: Gender of the customer (Male/Female).
- **Age**: Customer age in years. 
- **Annual Income ($)**: Customer annual income in USD.
- **Spending_Score (1-100)**: Customer spending score.
- **Profession**: Profession of the customer.
- **Work_Experience**: Work Experience in years.
- **Family_Size**: Number of family members for the customer (including the customer).

## Value Distribution

```
df['Gender'].value_counts()
```

<img width="140" height="67" alt="image" src="https://github.com/user-attachments/assets/59965c63-a1b3-44aa-a078-879a24c2ea9f" />

- Females = 59.3% of the dataset
- Males = 40.7% of the dataset

So the dataset is female‑dominant, but not overwhelmingly so. This suggests that purchasing behavior, preferences, and engagement patterns may be more strongly influenced by female consumers.

```
df['Profession'].value_counts(dropna=False)
```
<img width="172" height="229" alt="image" src="https://github.com/user-attachments/assets/0c43a075-77f6-4734-8ec9-131672010d1d" />

The largest groups are:
- Artists (612)
- Healthcare workers (339)
- Entertainment professionals (234)

This indicates a customer base heavily represented by expressive, lifestyle-driven, and service-focused occupations, each of which tends to have distinct motivations and spending tendencies.
#### Checking value distributions using histograms before preprocessing and cleaning. 
```
fig, ax = plt.subplots(figsize=(14,10))
df.drop(columns='CustomerID').hist(ax=ax)

plt.show()
```
<img width="1147" height="836" alt="image" src="https://github.com/user-attachments/assets/636ae6fe-5032-43ba-bdbd-87187adc7afa" />

From these histograms, it can be observed that:
- **Age**: The dataset spans all ages evenly without a dominant group.
- **Annual Income ($)**: Most individuals earn between $75K–$100K.
- **Spending_Score (1-100)**: Spending behavior is evenly distributed across the scale. 
- **Work_Experience**: Majority have 0 years; frequency drops steadily with more experience. 
- **Family_Size**: Small families of 1–3 members are most common, with larger households being less common.

#### Correlation Map
A correlation heat map is then used to visualize the relationships between variables. Correlation coefficients quantify the relationship between two variables, ranging from -1 to +1. 
- +1: Perfect positive correlation. When one variable increases, the other increases proportionally.
- 0: No linear relationship between the variables.
- -1: Perfect negative correlation. When one variable increases, the other decreases proportionally.
 
```
corr_map = df.drop(columns='CustomerID').corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_map, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heat Map")
plt.tight_layout()
plt.show()
```
<img width="930" height="790" alt="image" src="https://github.com/user-attachments/assets/79a8d896-6d4c-4e27-aafc-60a068c338d1" />

The heatmap does not reveal much. None of the variables show strong correlations (all values are close to 0). 

- **Age vs. Spending Score:** Slight negative correlation (-0.04). Older customers tend to spend a bit less, but the effect is weak.
- **Annual Income vs. Spending Score:** Very weak positive correlation (0.02). Income doesn’t strongly predict spending score.
- **Work Experience vs. Annual Income:** Small positive correlation (0.09). More experience is linked to slightly higher income.
- **Family Size vs. Annual Income:** Weak positive correlation (0.09). Larger families may have slightly higher income.
- **Age vs. Family Size:** Small positive correlation (0.04). Older individuals tend to have slightly larger families.

This suggests that spending behavior is not strongly driven by age, income, work experience, or family size in this dataset.

### Data Preprocessing and Cleaning
Before applying K-means, it is essential to prepare the data to ensure accurate and meaningful
results. There are 2,000 values on this dataset; of the 8 variables, 1 column (Profession) contains missing values.

<img width="253" height="276" alt="image" src="https://github.com/user-attachments/assets/a655692b-38ad-4734-896e-e4d41f4d0ded" />

Seeing as only 35 values are missing, which is 1.75% of the dataset, they were removed.

<img width="255" height="368" alt="Screenshot 2025-12-11 at 1 20 36 AM" src="https://github.com/user-attachments/assets/b997f216-752a-48d1-8d55-3374844b297e" />

### Data Transformation

K-means relies on distance-based measurements, so before applying K-Means clustering, the categorical values need to be transformed into a numerical representation. To start, the numerical and categorical values were seperated. 

```
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
```

Next, the categorical data was one-hot encoded. One Hot Encoding is a method for converting categorical variables into a binary format. It creates new columns for each category where 1 means the category is present and 0 means it is not. 

```
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded.info()
```
<img width="394" height="464" alt="image" src="https://github.com/user-attachments/assets/8a720b4b-2a12-4f6f-aa12-d6c35d60008a" />


The data was then normalized using standard scaler to reduce the impact of outliers and noise, making it easier for the algorithm to identify clusters.

```
X = StandardScaler().fit_transform(df_encoded)
```

## Cluster Development

To determine the proper number of clusters, the elbow method and sihlouette scores were utilized. The Elbow Method helps by plotting the Within-Cluster Sum of Squares (WCSS) against increasing k values and looking for a point where the improvement slows down, this point is called the "elbow." 

#### Elbow Method

```
wcss = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method For Optimal K')
plt.show()
```
<img width="713" height="470" alt="image" src="https://github.com/user-attachments/assets/22f8e8c8-6e9d-4c4f-9179-845eccc3e18c" />

#### The Silhouette Coefficient

The Silhouette Coefficient ranges from -1 (poor clustering) to 1 (well-separated clusters), with values near 0 indicating overlapping clusters.
```
k_values = range(2, 11) 

for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state = 0, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"Silhouette score for k={k}: {score:.2f}")
```
<img width="235" height="198" alt="image" src="https://github.com/user-attachments/assets/a755215e-1ba1-4558-b5c3-d320f8978690" />

### Setting up K-Means

The K-Means model was build after determining the optimal number of clusters. Then, the assigned cluster was then added into the dataframe.

```
clusterNum = 9
k_means = KMeans(n_clusters = clusterNum,random_state=0, n_init=10)
k_means.fit(X)
y = k_means.labels_

df['Cluster'] = y + 1
```
Checking the customers in each cluster.
```
df['Cluster'].value_counts(ascending=True)
```
<img width="86" height="230" alt="image" src="https://github.com/user-attachments/assets/706927e0-ea37-4a21-a91b-7eba2df894b3" />

## Analyzing Results

### Cluster Properties

```
cluster_profiles = df.groupby('Cluster').agg({
    'Gender': lambda x: x.mode()[0],
    'Age': 'mean',
    'Annual Income ($)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Profession': lambda x: x.mode()[0],
    'Work Experience': 'mean',
    'Family Size': 'mean'
}).reset_index().round(2)

print(cluster_profiles)
```

| Cluster | Gender |   Age  | Annual Income ($) | Spending Score | Profession    | Work Experience | Family Size |
|--------:|--------|--------|-------------------|----------------|---------------|------------------|-------------|
|       1 | Female | 51.16  | 110,650.33        | 52.94          | Entertainment | 3.50             | 3.89        |
|       2 | Female | 55.09  | 111,161.24        | 48.97          | Engineer      | 3.96             | 3.58        |
|       3 | Female | 47.84  | 112,574.04        | 50.52          | Healthcare    | 4.00             | 3.91        |
|       4 | Female | 47.75  | 110,995.84        | 48.86          | Lawyer        | 3.53             | 3.62        |
|       5 | Female | 49.00  | 108,776.58        | 52.68          | Artist        | 4.22             | 3.65        |
|       6 | Female | 46.62  | 111,573.22        | 51.90          | Doctor        | 4.30             | 3.67        |
|       7 | Female | 45.37  | 108,758.62        | 46.38          | Homemaker     | 6.13             | 4.05        |
|       8 | Female | 45.82  | 107,994.21        | 48.72          | Marketing     | 4.31             | 3.73        |
|       9 | Female | 46.60  | 113,770.13        | 49.90          | Executive     | 4.25             | 3.97        |

The output reflects a group of female customers within a consistent mid‑career age range and similar socioeconomic characteristics. Key observations include:

- The sample consists entirely of female customers.
- Ages range from 45 to 55, showing a mature demographic.
- Annual incomes are consistently high, ranging from approximately $108,000 to $114,000, reflecting strong purchasing power.
- Spending scores are grouped between 46 and 53, indicating balanced spending behavior.
- Professional backgrounds are diverse, reflecting a mix of creative, technical, and leadership professions.
- Most individuals have around 3.5 to 4.3 years of experience, while the Homemaker profile shows 6.13 years.
- Family sizes range from 3.58 to 4.05, indicating multi‑member households, potentially influencing spending priorities.

## Output Interpretation

Examining the average values of numeric variables for each cluster:
```
numeric_cols = df.select_dtypes(include=np.number).drop(['CustomerID', 'Cluster'], axis=1).columns
fig = plt.figure(figsize=(20, 20))

for i, column in enumerate(numeric_cols):
    df_plot = df.groupby('Cluster')[column].mean().reset_index()  
    ax = fig.add_subplot(5, 2, i+1)
    sns.barplot(x= 'Cluster',hue='Cluster', y=column, data=df_plot, palette='coolwarm', ax=ax,legend=False)
    ax.set_title(f'Average {column.title()} per Cluster')

    
plt.tight_layout()    
plt.show()
```
<img width="1989" height="1221" alt="image" src="https://github.com/user-attachments/assets/16f47046-0c92-4854-907d-a86f8ad96a1d" />

These bar charts show how each variable differs across clusters:
- Cluster 1 shows the highest spending score, closely followed by Cluster 5, suggesting these groups are the most active spenders.
- Cluster 2 has the highest average age, indicating this group represents the oldest customers.
- Cluster 7 stands out with the highest work experience and largest family size, making it the most established and family‑centric group.
- Cluster 9 earns the highest annual income, making it the most affluent segment.


#### Examining the distribution of categorical variables across clusters:
```
fig = plt.figure(figsize=(20, 20))

for i, column in enumerate(categorical_cols):
    df_plot = df.groupby(['Cluster', column]).size().reset_index(name='count')
    ax = fig.add_subplot(5, 2, i+1)
    sns.barplot(x='Cluster', y='count', hue=column, data = df_plot, palette='coolwarm',ax=ax)
    ax.set_title(f'Distribution of {column.title()} per Cluster')

    
plt.tight_layout()
plt.show()
```
<img width="1989" height="428" alt="image" src="https://github.com/user-attachments/assets/e5a7d2cf-1d34-4700-8b7d-b8a88affc327" />

These charts reveal:

- Cluster 3 is strongly defined by Healthcare professionals, making it a distinctly medical‑oriented segment.
- Cluster 5 is the largest and most gender‑populated group, it is overwhelmingly dominated by Artists.
- Cluster 7 is the smallest cluster in terms of gender representation, it is as a group with a limited population.
- Cluster 9 is mostly composed of Executives, it is the most corporate‑leaning and professionally senior segment in the dataset.

## Conclusion

### Summary

The segmentation analysis reveals a customer base that is consistently affluent, with average annual incomes ranging from $108K to $114K. Spending scores fall in the 46–53 range, showing moderate but steady consumption patterns. Demographically, most clusters are in their mid-to-late 40s, with family sizes averaging 3.6–4.0, and gender distribution generally balanced, though some clusters (like Artists) skew female. Profession is the strongest differentiator, with clear clusters across Entertainment, Engineers, Healthcare, Lawyers, Artists, Doctors, Homemakers, Marketing, and Executives.

#### Key Financial Insights

- Highest income: Executives (~$114K)
- Lowest income: Marketing (~$108K)
- Highest spending score: Entertainment (52.9) and Artists (52.7)
- Lowest spending score: Homemakers (46.4)


### Segmentation Profiles

| Cluster | Customer Segmentation | Key Traits | Marketing Implications |
|---------|-----------------------|------------|-------------------------|
| 1 | Entertainment | Mid-50s, balanced gender, high income, moderate spending | Creative, lifestyle-driven, family-oriented → Premium lifestyle products, exclusive events, entertainment bundles |
| 2 | Engineers | Oldest group (~55), high income, moderate spending | Analytical, stable, career-focused → Tech products, financial planning, retirement services |
| 3 | Healthcare | Late 40s, high income, balanced gender | Health-conscious, professional, family-oriented → Wellness products, healthcare tech, family insurance |
| 4 | Lawyers | Late 40s, smaller cluster, moderate spending | Professional, prestige-driven → Luxury services, networking events, productivity tools |
| 5 | Artists | Largest cluster, ~49 years, female-skewed | Creative, expressive, lifestyle-oriented → Artistic experiences, creative tools, loyalty programs |
| 6 | Doctors | Mid-40s, high income, moderate spending | Professional, health-focused, family-oriented → Premium healthcare, professional development, investment products |
| 7 | Homemakers | Smallest cluster, ~45 years, lowest spending | Family-centric, experienced → Value-driven promotions, household bundles, family packages |
| 8 | Marketing | Mid-40s, balanced gender, moderate spending | Trend-driven, aspirational → Digital-first campaigns, lifestyle branding, aspirational products |
| 9 | Executives | Mid-40s, high income, moderate spending | Leadership-oriented, prestige-driven → High-end services, networking, productivity-enhancing tools |

#### Strategic Opportunities

Premium offerings should be directed at **Entertainment, Artists, and Executives**, who combine high income or high spending scores with strong lifestyle orientation.  

**Homemakers and Lawyers** show lower spending scores and smaller cluster sizes, suggesting opportunities for tailored engagement through value-driven promotions or prestige services. 

**Healthcare and Doctors** present overlapping professional interests, making wellness and professional development campaigns effective.  

**Engineers and Executives** represent high-income, career-focused segments where financial planning and productivity tools can unlock latent value.

Overall, the segmentation highlights a **high-income but moderate-spending customer base**. By aligning marketing strategies with professional identity, age, and spending behavior, companies can capture both **short-term gains through targeted promotions** and **long-term loyalty through lifestyle and professional alignment**.
