# baseball_ballistics_clf
Repository for my General Assembly Data Science Immersive Capstone Project - predicting hit or no hit based on the ballistics of each individual pitch and ball hit in play.

## Executive Summary

### Problem Statement
MLB Advanced Media, as stated in a job description for which I was intrigued by, was looking to develop insights into predictability of a hit based on data acquired through their Statcast tool. Statcast is a high-speed, high-accuracy device that tracks ball and player movements. 

The findings of this task would be for use by analysts and commentators during game broadcasts. The problem statement for the specific prediction I undertook is:

Based on the ballistics of the pitch and the ball hit into play, what is the likelihood it results in a hit.


### Metrics / Feature Information
**Target:** `hit_flag` == `True` or `False` 

**Predictors:** 
`mph` == speed in miles per hour of pitch thrown
`ev_mph` == speed in miles per hour of ball hit off the bat
`dist` == distance in feet the ball traveled off the bat
`spin_rate` == the rate at which the ball pitched is spinning
`launch_angle` == the angle at which the ball comes off the bat
`zone` == the section of the strikezone that the ball crosses when it is hit
`ab_count` == encoding represented as '#-#' indicating the number of balls and strikes the at-bat as accrued prior to the ball being hit in play
`inning` == the inning of the game when the ball was hit
`full_pitch` == the specific type of the pitch thrown
`pitch_rollup` == a higher-level indication of the type of pitch thrown

**Full Data (after data clean-up):**
Index: 127052 entries, 434378-8 to 621056-1
Data columns (total 18 columns):
player_id       127052 non-null int64
pitch           127052 non-null object
mph             127052 non-null float64
ev_mph          127052 non-null float64
pitcher         127052 non-null object
batter          127052 non-null object
dist            127052 non-null int64
spin_rate       127052 non-null int64
launch_angle    127052 non-null float64
zone            127052 non-null object
game_date       127052 non-null object
ab_count        127052 non-null object
inning          127052 non-null object
pitch_result    127052 non-null object
ab_result       127052 non-null object
full_pitch      127052 non-null object
pitch_rollup    127052 non-null object
hit_flag        127052 non-null bool
dtypes: bool(1), float64(3), int64(3), object(11)
memory usage: 17.6+ MB

`zone` categories explained in attached .png:
![alt text](https://github.com/aigruber22/baseball_ballistics_clf/blob/master/zone.png)

### Findings


### Risks / Assumptions / Limitations
Limited fields to select from pertaining to ballistics, potential risk in predictability of hit.


## Steps
1. **Scrape** baseballsavant.mlb.com for pitch-level statcast data from the 2017 season and **store pitch data in postgres**
- Player-level data available at url source of data as .csv. Imported those two files to map in `player_id`.
- Defined series of functions that are subsequently imbedded in each other. In running them on the `player_id`, the fucntions scrape the data from the url and store them in a Postgres database.
- Scraping is performed with `XPath` querying, parsing the html from the website to pull pertinent data. `sqlalchemy` used in tandem with PostgreSQL to store data.
- `721,436` total pitches scraped from baseballsavant.mlb.com. These are all the pitches thrown during the 2017 MLB regular season.

`url = 'https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&\
    hfNewZones=&hfGT=R%7C&hfC=&hfSea=2017%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&\
    batter_stands=&hfSA=&game_date_gt=&game_date_lt=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&\
    min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&\
    min_abs=0&type=details&player_id={}'.format(player_id)`

2. **Clean data** by handling null values and adjusting data types and **engineer features** necessary for analysis
- Read in and inspect data from Postres
- Inspect null-values, determine null definitions, and incorporate null-handling procedures

|   Fields   |   Null Meaning  |   Data Type   |
| -----------|:---------------:|--------------:|
| pitch | unknown | Object |
| mph | unknown | Float |
| ev_mph | ball not hit | Float |
| dist | ball not hit | Integer |
| spin_rate | unknown | Integer |
| launch_angle | ball not hit | Float |
| zone | unknown | Object |
| ab_result | pitch did not end ab | Object | 

- Correct data types
- Merge in pitch types to provide clarity to the abbreviated field `pitch`
- Using `pitch_results` and `ab_result`, create `hit_flag` field to be used as modeling target
- Pickle data

3. Perform **EDA** to understand the data more deeply
- `.describe()` to view basic statistics of the data set
- Visualizations
    - `sns.pairplot`
    - `plt.hist` with KDE line, mean, and median plotted
    - `violinplot`
    - `sns.heatmap` to show correlation between features
    - Bar plots on counts for categorical features, split by hit / no hit

4. Calculate a **benchmark metric** to understand accuracy by guessing same target for every row
- A guess of no hit for every ball in play results in 67.4% accuracy and an F1 score of .805

5. Further **feature engineering** by **one-hot encoding** categorical features
- To support modeling, categorical features are one-hot encoded with `pandas.get_dummies()`.

6. Perform **benchmark models** on data without preprocessing and feature selection on the data and without hyperparameter tuning on the model
- Define function to train and test a model using specified predictors and targets
- Models will be trained without any preprocessing or feature selection on the data and without any hyperparameter tuning in the model to get a benchmark accuracy score
- Models used:
    - K Nearest Neighbors
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
- Results:
    - Random Forest and K Neighbors performed best, each predicting with ~80% accuracy on the test data. Compared to the ~67% by guessing no hit for everything, this is good but we can do better.
    - *Note: Despite high train scores, the tree models are not overfit. The test score is pretty close to the other two models' test scores.*

|   Model Name   |   Test Score  |   Train Score   |
| -----------|:---------------:|--------------:|
| K Nearest Neighbors | 0.7922 | 0.8530 |
| Logistic Regression | 0.7196 | 0.7182 |
| Decision Tree Classifier | 0.7544 | 1.0 |
| Random Forest Classifier | 0.7987 | .9871 |

7. **Normalize** / **Standardize** data and run same models on that data and assess scores
- Define function to train and test a model using specified predictors and targets
- Apply `MinMaxScaler()` to data
- Train model on normalized data
- Models used:
    - K Nearest Neighbors
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
- Results:
    - While logistic regression, decision tree, and random forest perform the same with `MinMaxScaler` normalization as they did before, K neighbors performs significantly worse. `MinMaxScaler` is forcing every feature to a range between 0 and 1. 

    - Based on EDA, I believe there are some very unimportant features in the data that have become more noisy as a result of normalization. However, my hypothesis is this will make it easier to parse them out in feature selection later on.

|   Model Name   |   Test Score  |   Train Score   |
| -----------|:---------------:|--------------:|
| K Nearest Neighbors | 0.6662 | 0.7797 |
| Logistic Regression | 0.7209 | 0.7204 |
| Decision Tree Classifier | 0.7563 | 1.0 |
| Random Forest Classifier | 0.8011 | 0.9874 |

8. Add **deskewing** method and check score of models again
- Define function to train and test a model using specified predictors and targets
- Deskew data using `box_cox()`
- Apply `MinMaxScaler()` to data
- Train model on deskewed and normalized data
- Models used:
    - K Nearest Neighbors
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
- Results:
    - With addition of `box_cox` skew-normalization, not much changes from findings on `MinMaxScaler` alone.

|   Model Name   |   Test Score  |   Train Score   |
| -----------|:---------------:|--------------:|
| K Nearest Neighbors | 0.6621 | 0.7784 |
| Logistic Regression | 0.7516 | 0.7465 |
| Decision Tree Classifier | 0.7571 | 1.0 |
| Random Forest Classifier | 0.7972 | 0.9875 |

9. **Select important featurs** and then rerun models again.
- Deskew data using `box_cox()`
- Apply `MinMaxScaler()` to data
- Using `SelectFromModel()` with a `LogisticRegression()` estimator and an L1 penalty, select important features
- Define function to train and test a model using specified predictors and targets
- Train model on deskewed, normalized, and feature optimized data
- Models used:
    - K Nearest Neighbors
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
- Results:
    - As suspected, while normalizing the data in the earlier step drastically lowered the test score for kneighbors, after applying feature selection the test accuracy score has now improved to around 80% with significantly lower dimensionality. Random Forest is also around 80% accuracy.

|   Model Name   |   Test Score  |   Train Score   |
| -----------|:---------------:|--------------:|
| K Nearest Neighbors | 0.8005 | 0.8596 |
| Logistic Regression | 0.7451 | 0.7485 |
| Decision Tree Classifier | 0.7638 | 0.9998 |
| Random Forest Classifier | 0.8010 | 0.9834 |

10. **Resample** classes so our target classes (hit, no_hit) are balanced.
- Resample target class
- Deskew data using `box_cox()`
- Apply `MinMaxScaler()` to data
- Using `SelectFromModel()` with a `LogisticRegression()` estimator and an L1 penalty, select important features
- Train model on deskewed, normalized, and feature optimized data
- Define function to train and test a model using specified predictors and targets
- Models used:
    - K Nearest Neighbors
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
- Results:
    - Test scores drop ever so slightly after resampling the target classes. This suggests that the accuracy score was getting bolstered by random chance. While the model doesn't predict as accurately, it is more represntative of how it'll perform on out of sample data.

|   Model Name   |   Test Score  |   Train Score   |
| -----------|:---------------:|--------------:|
| K Nearest Neighbors | 0.7792 | 0.8456 |
| Logistic Regression | 0.6989 | 0.6943 |
| Decision Tree Classifier | 0.7407 | 0.9998 |
| Random Forest Classifier | 0.7783 | 0.9838 |

11. Run each model through a 10-fold **cross-validation** and score each model.
- After all preprocessing steps, cross-validation confirms our findings from our original benchmarking...K Neighbors and Random Fores are the best performers for this data. Since K Neighbors scores really slowly, for the remaining notebooks only Random Forest will be used.

12. Using one of the stronger performing models from prior steps, attempt to **tune hyperparameters** to improve model performance.
**Model:** 
`RandomForestClassifier()`

**Hyperparameters tuning:** 
n_estimators, criterion

**Results:**
Best parameters: n_estimators = 9, criterion = 'gini'
Accuracy Score = ~77%

## Possible Next Steps
1. Tune more / different hyperparameters
2. Examine PCA as an option on the data set instead of feature selection


## Statistical Approach
**Models:** K Neighbors Classifier, Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
**Implemention:** Deskew -> Normalize -> Feature Selection -> Resample -> Model with benchmarking along the way.
**Evaluation:** Accuracy Score, Cohen Kappa Score

## Key Findings
- Violion plots show hitting ballistics to differ between the two classes, while pitching ballistics look the same across the two classes
- Feature Selection indicated 6 important features:
    - `ev_mph`, `dist`, `launch_angle`, `zone_13.0`, `zone_14.0`, `zone_unknown`
    - This validates the above point that hitting ballistics are more important in predicting hit / no hit as compared to the pitch ballistics. All types of pitches at different speeds with different spin rates are balanced across the two classes. This means that a players ability to read or guess the pitch and then execute the swing is more important than the actual ballistics of the pitch in determining a hit or no hit.
- Of all balls hit in play in 2017, ~67% of them were not a hit
- Using K Nearest Neighbors or Random Forest Classifier on the full data set, we can predict with ~80% accuracy whether a particular pitch / ball in play is a hit or not. When we resample so the baseline is 50%, we can predict with ~77% accuracy with those same models. I believe this to be the expected accuracy on out of sample data.


## Modeling Notes
- Preprocessing did little to affect the predictive accuracy of the model
- Resampling done towards the end of the workflow to attempt to mitigate predictive score being boosted by random chance / lucky guesses
    
