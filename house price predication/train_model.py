#Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"C:\Users\HP\AppData\Local\Temp\Rar$DIa0.441\House_data.csv")



#. Data Shape,Structure and Types



dataset.shape

dataset.head() 

dataset.tail() 

dataset.info() 

dataset.dtypes

dataset.index


def audit(dataset):
    return pd.DataFrame(
        {
            'dtype'    :  dataset.dtypes,
            'nulls'    :  dataset.isnull().sum(),
            'null_%'   :  (dataset.isnull().mean() * 100).round(2),
            'unique'   :  dataset.nunique(),
            'unique_%' :  (dataset.nunique().mean() * 100).round(2),
            'zeros'    :  (dataset == 0).sum(),
            'sample'   :   dataset.iloc[0]
        }
    )

audit(dataset)

# Descriptive Statistics

dataset.describe()


plt.figure(figsize=(18, 15))
corr_data = dataset.drop(['date','price','id'], axis=1)


corr_mask = np.triu(corr_data.corr())
h_map = sns.heatmap(corr_data.corr(), mask=corr_mask, cmap='Blues')
h_map


features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot',
            'floors','sqft_above','sqft_basement']


fig, axes = plt.subplots(3,3, figsize=(15,12))
axes = axes.flatten()


for i, feature in enumerate(features):
    sns.histplot(dataset[feature], kde=True, ax=axes[i], color='green')
    axes[i].set_title(f'{feature} Distribution')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')
    axes[i].grid(True, linestyle='--', alpha=0.5)
    
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])
                
plt.tight_layout()
plt.show()





# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns

# First plot: Number of Bedrooms
sns.countplot(ax=axes[0], x='bedrooms', data=dataset, palette='Greens_d')
axes[0].set_title("Number of Bedrooms")
axes[0].set_xlabel("Bedrooms")
axes[0].set_ylabel("Count")
axes[0].grid(axis='y', linestyle="--", alpha=0.5)


# Second plot: Number of Bathrooms
sns.countplot(ax=axes[1], x='bathrooms', data=dataset, palette='Set2')
axes[1].set_title("Number of Bathrooms")
axes[1].set_xlabel("Bathrooms")
axes[1].set_ylabel("Count")
axes[1].grid(axis='y', linestyle="--", alpha=0.5)

# Adjust layout
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='condition', y='price', data=dataset, palette='coolwarm')
plt.title("Price vs. Condition")
plt.xlabel("Condition (1 = Worst, 5 = Best)")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# pairplot
sns.pairplot(dataset[['price', 'bedrooms', 'bathrooms', 'sqft_living']])
plt.show()



features = ['price', 'sqft_lot']

# Set up the subplot grid (3 rows x 3 columns to fit 7 plots neatly)
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot distributions
for i, feature in enumerate(features):
    sns.histplot(dataset[feature], kde=True, ax=axes[i], color='green')
    axes[i].set_title(f'{feature} Distribution')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Hide unused subplots (if any)
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



dataset.head()


# Target variable
y = dataset['price']

# Features (drop unnecessary columns)
x = dataset.drop(['price', 'id', 'date'], axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2 , random_state = 40)

print("shape of X_train = ", x_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", x_test.shape)
print("shape of y_test = ", y_test.shape)

x_train

x_test.shape

y_train
y_test

from sklearn.model_selection import train_test_split

# Features
x = dataset.drop(['price', 'id', 'date'], axis=1)

# Target
y = dataset['price']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=40
)

print("Shape of X_train :", x_train.shape)
print("Shape of X_test  :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test  :", y_test.shape)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

import pickle

# Save model
pickle.dump(model, open("house_model.pkl", "wb"))

print("Model Saved Successfully")