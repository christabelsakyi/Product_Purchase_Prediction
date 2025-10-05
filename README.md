## Product Purchase Prediction

### **Exercise: Building and Optimizing a Decision Tree Classifier for Product Purchase Prediction**

As a newly hired AI Engineer, your task is to predict customer behavior based on various features such as age, income, and gender. This exercise involves cleaning the data, training a decision tree model, and evaluating the model's performance to understand the key factors influencing customer purchasing decisions.

### **Dataset**

You are provided with a dataset that contains customer information and their product purchase behavior.

**Dataset columns:**

* `Age`: The age of the customer.  
* `Income`: The income of the customer (in thousands).  
* `Gender`: The gender of the customer (Male/Female).  
* `Buy_Product`: The target variable indicating whether the customer bought the product (1 for yes, 0 for no).

---

### **Instructions**

#### **1\. Data Preprocessing**

1. **Load the dataset** from the provided CSV file.  
2. **Convert categorical variables**: Convert the `Gender` column from categorical values (`Male`, `Female`) to numerical format (`Male = 0`, `Female = 1`).  
3. **Normalize the features**: Apply Min-Max scaling to the `Age` and `Income` columns to bring the values between 0 and 1\.  
4. **Split the dataset** into:  
   * Features (`X`): `Age`, `Income`, and `Gender`.  
   * Target variable (`y`): `Buy_Product`.

#### **2\. Train a Decision Tree Classifier**

1. **Train a Decision Tree model** using the features (`Age`, `Income`, and `Gender`) to predict `Buy_Product`.  
2. **Hyperparameter tuning**: Use Grid Search to optimize the model's hyperparameters such as `max_depth` and `min_samples_split`. Report the best parameters.

#### **3\. Make Predictions**

1. **Predict** the target for the following new data points:  
   * `Age = 40`, `Income = 50`, `Gender = Male`  
   * `Age = 30`, `Income = 45`, `Gender = Female`

#### **4\. Evaluate the Model**

1. **Calculate the model's accuracy** on the entire dataset.  
2. **Generate the confusion matrix** and **classification report** to evaluate the model’s performance.

#### **5\. Visualize the Decision Tree**

1. **Plot the decision tree** to understand how the model makes predictions. Include feature names and class labels in the visualization.

#### **6\. Cross-Validation**

1. Perform **5-fold cross-validation** on the model and report the average accuracy score.

---

### **Deliverables**

1. **Preprocessed Dataset**: Include the dataset after preprocessing (with normalized features and encoded gender).  
2. **Trained Model**: Submit the decision tree model after training and hyperparameter optimization.  
3. **Model Evaluation**: Provide the accuracy, confusion matrix, and classification report.  
4. **Visualizations**: Include a plot of the decision tree.  
5. **Cross-Validation Results**: Report the average accuracy score from 5-fold cross-validation.


