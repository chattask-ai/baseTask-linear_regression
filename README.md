# Linear Regression

## Introduction

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to find a linear equation that best predicts the dependent variable based on the independent variables.

### Mathematical Formulation

For a simple linear regression with one independent variable, the relationship is modeled as:

\[ y = \beta_0 + \beta_1 x + \epsilon \]

where:
- \( y \) is the dependent variable
- \( x \) is the independent variable
- \( \beta_0 \) is the intercept
- \( \beta_1 \) is the slope (regression coefficient)
- \( \epsilon \) is the error term

For multiple linear regression with multiple independent variables, the relationship is:

\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon \]

where \( x_1, x_2, \ldots, x_n \) are the independent variables.

## Process of Linear Regression

### Using Python

#### 1. Load Data

First, load your data into a pandas DataFrame.

```python
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
```

#### 2. Fit the Linear Regression Model

Using `scikit-learn`, fit a linear regression model to the data.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define independent variables (X) and dependent variable (y)
X = data[['independent_var1', 'independent_var2']]  # Replace with your variables
y = data['dependent_var']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

#### 3. Evaluate the Model

Evaluate the performance of the model using metrics such as R-squared and Mean Squared Error (MSE).

```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```

#### 4. Visualize the Results

Visualize the actual vs. predicted values.

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
```

### Using R

#### 1. Load Data

First, load your data into an R dataframe.

```r
library(readr)

# Load your data
data <- read_csv('your_data.csv')
```

#### 2. Fit the Linear Regression Model

Using the `lm` function, fit a linear regression model to the data.

```r
# Fit the model
model <- lm(dependent_var ~ independent_var1 + independent_var2, data = data) # Replace with your variables

# Summary of the model
summary(model)
```

#### 3. Evaluate the Model

Evaluate the performance of the model using metrics such as R-squared and Mean Squared Error (MSE).

```r
# Calculate R-squared
r_squared <- summary(model)$r.squared

# Calculate Mean Squared Error
mse <- mean(model$residuals^2)

print(paste('R-squared:', r_squared))
print(paste('Mean Squared Error:', mse))
```

#### 4. Visualize the Results

Visualize the actual vs. predicted values.

```r
library(ggplot2)

# Predict values
data$predicted <- predict(model, data)

# Plot actual vs predicted values
ggplot(data, aes(x = dependent_var, y = predicted)) +
  geom_point() +
  labs(x = 'Actual Values', y = 'Predicted Values', title = 'Actual vs Predicted Values') +
  theme_minimal()
```

## Conclusion

Linear regression is a powerful and simple method for modeling relationships between variables. By understanding the theory and how to implement it using tools like Python and R, you can effectively analyze and predict outcomes based on your data.
