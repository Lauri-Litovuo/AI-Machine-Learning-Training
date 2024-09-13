# AI/ML Piscine - Basics and Advanced Modules

This repository contains my journey through both the **Basic** and **Advanced AI/ML Piscines** at Hive Helsinki Coding School. These intensive programs introduced foundational and advanced concepts of machine learning, focusing on key regression and classification techniques, while using Python’s powerful libraries like **Scikit-learn**, **Pandas**, **NumPy**, and others. I utilized **Google Colab** as the platform for developing and running these projects.

## Table of Contents
- [Overview](#overview)
- [Basic AI/ML Piscine](#basic-ai-ml-piscine)
  - [Day 00: Simple Linear Regression](#day-00-simple-linear-regression)
  - [Day 01: Simple Linear Regression 2 (Numpy)](#day-01-simple-linear-regression-2-numpy)
  - [Day 02: Multiple Linear Regression](#day-02-multiple-linear-regression)
- [Advanced AI/ML Piscine](#advanced-ai-ml-piscine)
  - [Day 00: Binary Classification with Logistic Regression](#day-00-binary-classification-with-logistic-regression)
  - [Day 01: Multinomial Logistic Regression](#day-01-multinomial-logistic-regression)
  - [Day 02: Classification with Other Models (Decision Trees and XGBoost)](#day-02-classification-with-other-models-decision-trees-and-xgboost)
- [Example Project - Advanced Piscine Day 02](#example-project---advanced-piscine-day-02)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)

## Overview

Throughout this repository, you will find my work on both the **Basic** and **Advanced** AI/ML Piscines. The basic module focused on core regression techniques, while the advanced module covered various classification algorithms and their applications. I completed all projects using **Google Colab**, which provided a flexible environment for running and experimenting with the code.

---

## Basic AI ML Piscine

The **Basic AI/ML Piscine** was designed to introduce core concepts in regression, with a focus on hands-on implementation using key Python libraries such as **Pandas**, **NumPy**, and **Matplotlib**. This section covers simple and multiple linear regression models.

### Day 00: Simple Linear Regression

On Day 00, I explored the basics of **simple linear regression**, focusing on applying it to real-world datasets. Using libraries like **Pandas** for data manipulation and **Matplotlib** for visualization, I gained practical experience in building and interpreting linear models.

#### Key Learnings:
- Understanding the principles of **simple linear regression**.
- Manipulating datasets using **Pandas**.
- Visualizing regression results with **Matplotlib**.
- Using **Scikit-learn** to implement and evaluate linear regression models.

#### Tools:
- **Pandas**: For data manipulation.
- **Matplotlib**: For visualizing the regression line and residuals.
- **Scikit-learn**: For model implementation and evaluation.

---

### Day 01: Simple Linear Regression 2 (Numpy)

Building on the concepts from Day 00, Day 01 took a different approach by implementing **simple linear regression** using **NumPy** for mathematical computations, rather than relying on high-level libraries like Scikit-learn. This deepened my understanding of the mathematics behind regression.

#### Key Learnings:
- Manually computing regression parameters using **NumPy**.
- Strengthening my understanding of the **gradient method** for optimization.
- Implementing regression models without using **Scikit-learn**, focusing on math-centric methods.

#### Tools:
- **NumPy**: For mathematical operations, including calculating regression coefficients and predictions.
- **Matplotlib**: For visualizing regression lines and data points.

---

### Day 02: Multiple Linear Regression

On Day 02, I expanded the concepts of simple linear regression to **multiple linear regression**, where several independent variables are used to predict a dependent variable. This module introduced more complexity and allowed me to explore richer predictive models.

#### Key Learnings:
- Understanding the theory behind **multiple linear regression**.
- Manipulating datasets with multiple variables using **Pandas**.
- Visualizing the relationships between variables with **Matplotlib** and **Seaborn**.
- Implementing models that predict based on several inputs.

#### Tools:
- **Pandas**: For handling multi-variable datasets.
- **Matplotlib/Seaborn**: For visualizing multiple relationships and regression results.
- **NumPy**: For calculating regression coefficients.
- **Scikit-learn**: For comparison and evaluation of models.

---

## Advanced AI ML Piscine

The **Advanced AI/ML Piscine** extended the learning from the basic module, with a focus on **classification** tasks using models like **logistic regression**, **decision trees**, and **XGBoost**. The advanced module introduced more complex algorithms and techniques for solving machine learning problems.

### Day 00: Binary Classification with Logistic Regression

The first day of the advanced module focused on **binary classification** using **logistic regression**, where I explored how to predict binary outcomes (e.g., yes/no) with logistic models.

#### Key Learnings:
- Building and interpreting **logistic regression** models.
- Understanding classification metrics like **accuracy**, **precision**, and **recall**.
- Implementing models using **Scikit-learn** and visualizing outcomes with **Matplotlib**.

#### Tools:
- **Scikit-learn**: For building logistic regression models.
- **Pandas**: For preprocessing datasets.
- **Matplotlib/Seaborn**: For visualizing decision boundaries and results.

---

### Day 01: Multinomial Logistic Regression

On the second day, I extended logistic regression to **multiclass classification**, learning how to predict outcomes across more than two categories.

#### Key Learnings:
- Understanding **multinomial logistic regression** and its applications.
- Handling multiclass data and applying appropriate preprocessing techniques.
- Evaluating multiclass models using metrics like the **confusion matrix** and **classification report**.

#### Tools:
- **Scikit-learn**: For multinomial logistic regression.
- **Pandas**: For handling multiclass datasets.
- **Matplotlib**: For visualizing class separations.

---

### Day 02: Classification with Other Models (Decision Trees and XGBoost)

The final day of the advanced module introduced more powerful classification algorithms: **Decision Trees** and **XGBoost**. These models are widely used for real-world classification problems due to their performance and interpretability.

#### Key Learnings:
- Building and interpreting **decision tree** models.
- Implementing **XGBoost**, a high-performance gradient boosting model.
- Evaluating models using cross-validation and fine-tuning hyperparameters.

#### Tools:
- **Scikit-learn**: For building and evaluating decision trees.
- **XGBoost**: For high-performance classification tasks.
- **Pandas**: For preparing datasets.
- **Matplotlib/Seaborn**: For visualizing decision boundaries and model results.

---

## Example Project - Advanced Piscine Day 02

Here’s an overview of the **Day 02** project from the advanced piscine, focusing on classification with other models. See [Day02 Jypiter Notebook](https://github.com/Lauri-Litovuo/AI-Machine-Learning-Training/blob/main/Medium02.ipynb).

### Project Description

For this project, I worked with an unbalanced banking dataset to classify financial transactions as fraudulent or legitimate. The exercise involved several key steps:

1. **Data Exploration**: I loaded the data into Google Colab and performed exploratory data analysis to understand the dataset.
2. **Feature Analysis**: I examined the relationships between numerical attributes and their influence on transaction outcomes, using visualization techniques.
3. **Handling Imbalance**: Given the unbalanced nature of the dataset, I applied rebalancing techniques to address the class imbalance.
4. **Model Evaluation**: I initially tested logistic regression, achieving a relatively poor accurancy. To improve performance, I explored more powerful models and aimed to achieve an accuracy greater than 0.98.

#### Key Learnings:
- Understanding and handling class imbalance in datasets.
- Using visualization techniques to explore and understand data.
- Implementing and comparing multiple classification models to achieve high accuracy.

#### Tools:
- **Google Colab**: Platform for executing and analyzing the project.
- **Scikit-learn**: For implementing and evaluating classification models.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For visualizing data and model performance.

---

## Technologies Used

This project leveraged the following key libraries and tools:

- **Python 3.x**
- **Scikit-learn**: For building and evaluating machine learning models.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For mathematical computations and implementing regression from scratch.
- **Matplotlib/Seaborn**: For data visualization and model interpretation.
- **XGBoost**: For gradient boosting models, providing powerful classification solutions.
- **Google Colab**: For developing and running the projects.

---

## Conclusion

The **AI/ML Piscine** programs provided a solid foundation for both basic and advanced machine learning concepts, focusing on practical, hands-on learning. From understanding linear regression to exploring more complex models like decision trees and XGBoost, I gained valuable skills for applying machine learning techniques to real-world data.

These Piscines have equipped me with the tools and knowledge necessary to solve a wide variety of regression and classification problems, and I look forward to expanding these skills further in future projects.
