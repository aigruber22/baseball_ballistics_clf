# baseball_ballistics_clf
Repository for my General Assembly Data Science Immersive Capstone Project - predicting hit or no hit based on the ballistics of each individual pitch and ball hit in play.

## Executive Summary

### Problem Statement
MLB Advanced Media, as stated in a job description for which I was intrigued by, was looking to develop insights into probability and prediction of a hit based on data acquired through their Statcast tool. Statcast is a high-speed, high-accuracy device that tracks ball and player movements. This is information intended to be used by analysts and commentators during game broadcasts. The problem statement for the specfic prediction I undertook is:

Based on the ballistics of the pitch and the ball hit into play, what is the likelihood it results in a hit.


### Metrics
Target: `hit_flag` == `True` or `False` 

### Findings


### Risks / Assumptions / Limitations
Limited fields to select from pertaining to ballistics, potential risk in predictability of hit.


## Steps
1. **Scrape** baseballsavant.mlb.com for pitch-level statcast data from the 2017 season and **store pitch data in postgres**
2. **Clean data** by handling null values and adjusting data types and **engineer features** necessary for analysis
3. Perform **EDA** to understand the data at a deeper level
4. Calculate a **benchmark metric** to understand accuracy by guessing same target for every row
5. Further **engineer features** by **one-hot encoding** categorical features
6. Perform **benchmark models** on data without preprocessing and feature selection
7. **Normalize** data and run same models on that data and assess scores
8. Add **deskewing** method and check score of models again
9. To be continued...

## Remaining Steps
1. Deskew-normalize the data
2. Examine PCA as an option
3. Build pipelines to model with all preprocessing steps
4. Look at power house models (boosting, random forest)
5. Package findings 


## Statistical Summary
**Models:** K Neighbors Classifier, Logistic Regression, Decision Tree Classifier. With extra time, I might try Boosting or Random Forest.
**Implemention:** Deskew -> Normalize -> Feature Selection -> Model with benchmarking along the way.
**Evaluation:**