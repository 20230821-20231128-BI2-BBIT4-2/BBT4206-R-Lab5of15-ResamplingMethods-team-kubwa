Business Intelligence Lab Submission Markdown
================
<Specify your group name here>
<Specify the date when you submitted the lab>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Loading the PimaIndians Dataset](#loading-the-pimaindians-dataset)
  - [Description of the Dataset](#description-of-the-dataset)
- [\<Splitting the dataset\>](#splitting-the-dataset)
  - [\<Classification: SVM with Repeated k-fold Cross
    Validation\>](#classification-svm-with-repeated-k-fold-cross-validation)
  - [\<Test the trained SVM model using the testing
    dataset\>](#test-the-trained-svm-model-using-the-testing-dataset)
  - [\<View a summary of the model and view the confusion
    matrix\>](#view-a-summary-of-the-model-and-view-the-confusion-matrix)

# Student Details

|                                                   |                                                                                                                                                                                                                                                                                                                                                                |     |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** | *\<list one Student name, class group (just the letter; A, B, or C), and ID per line, e.g., 123456 - A - John Leposo; you should be between 2 and 5 members per group\>* \| \| 1. 128998 - B - Crispus Nzano \| \| 2. 134100 - B - Timothy Obosi \| \| 3. 134092 - B - Alison Kuria \| 4. 135269 - B - Clifford Kipchumba \| \| 5. 112826 - B - Matu Ngatia \| |     |
| **GitHub Classroom Group Name**                   | Team Kubwa \|                                                                                                                                                                                                                                                                                                                                                  |     |
| **Course Code**                                   | BBT4206                                                                                                                                                                                                                                                                                                                                                        |     |
| **Course Name**                                   | Business Intelligence II                                                                                                                                                                                                                                                                                                                                       |     |
| **Program**                                       | Bachelor of Business Information Technology                                                                                                                                                                                                                                                                                                                    |     |
| **Semester Duration**                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023                                                                                                                                                                                                                                                                                                   |     |

# Setup Chunk

We start by installing all the required packages

``` r
## formatR - Required to format R code in the markdown ----

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# Resampling methods are techniques that can be used to improve the performance
# and reliability of machine learning algorithms. They work by creating
# multiple training sets from the original training set. The model is then
# trained on each training set, and the results are averaged. This helps to
# reduce overfitting and improve the model's generalization performance.

# Resampling methods include:
## Splitting the dataset into train and test sets ----
## Bootstrapping (sampling with replacement) ----
## Basic k-fold cross validation ----
## Repeated cross validation ----
## Leave One Out Cross-Validation (LOOCV) ----

# STEP 1. Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

------------------------------------------------------------------------

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following “*R Markdown*” options have been set as the
defaults in this markdown:

> output:  
>   
> github_document:  
> toc: yes  
> toc_depth: 4  
> fig_width: 6  
> fig_height: 4  
> df_print: default  
>   
> editor_options:  
> chunk_output_type: console

# Loading the PimaIndians Dataset

The PimaIndians Dataset is then loaded.

``` r
require("mlbench")
```

    ## Loading required package: mlbench

``` r
data("PimaIndiansDiabetes")

summary(PimaIndiansDiabetes)
```

    ##     pregnant         glucose         pressure         triceps     
    ##  Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :23.00  
    ##  Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54  
    ##  3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##     insulin           mass          pedigree           age        diabetes 
    ##  Min.   :  0.0   Min.   : 0.00   Min.   :0.0780   Min.   :21.00   neg:500  
    ##  1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437   1st Qu.:24.00   pos:268  
    ##  Median : 30.5   Median :32.00   Median :0.3725   Median :29.00            
    ##  Mean   : 79.8   Mean   :31.99   Mean   :0.4719   Mean   :33.24            
    ##  3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262   3rd Qu.:41.00            
    ##  Max.   :846.0   Max.   :67.10   Max.   :2.4200   Max.   :81.00

``` r
str(PimaIndiansDiabetes)
```

    ## 'data.frame':    768 obs. of  9 variables:
    ##  $ pregnant: num  6 1 8 1 0 5 3 10 2 8 ...
    ##  $ glucose : num  148 85 183 89 137 116 78 115 197 125 ...
    ##  $ pressure: num  72 66 64 66 40 74 50 0 70 96 ...
    ##  $ triceps : num  35 29 0 23 35 0 32 0 45 0 ...
    ##  $ insulin : num  0 0 0 94 168 0 88 0 543 0 ...
    ##  $ mass    : num  33.6 26.6 23.3 28.1 43.1 25.6 31 35.3 30.5 0 ...
    ##  $ pedigree: num  0.627 0.351 0.672 0.167 2.288 ...
    ##  $ age     : num  50 31 32 21 33 30 26 29 53 54 ...
    ##  $ diabetes: Factor w/ 2 levels "neg","pos": 2 1 2 1 2 1 2 1 2 2 ...

## Description of the Dataset

We then display the number of observations and number of variables. 9
Variables and 768 observations.

``` r
dim(PimaIndiansDiabetes)
```

    ## [1] 768   9

Next, we display the quartiles for each numeric
variable<span id="highlight" style="color: blue">*… this is the process
of **“storytelling using the data.”** The goal is to analyse the
PimaIndians dataset and try to train a model to make predictions( which
model is most suited for this dataset).*</span>

``` r
summary(PimaIndiansDiabetes)
```

    ##     pregnant         glucose         pressure         triceps     
    ##  Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :23.00  
    ##  Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54  
    ##  3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##     insulin           mass          pedigree           age        diabetes 
    ##  Min.   :  0.0   Min.   : 0.00   Min.   :0.0780   Min.   :21.00   neg:500  
    ##  1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437   1st Qu.:24.00   pos:268  
    ##  Median : 30.5   Median :32.00   Median :0.3725   Median :29.00            
    ##  Mean   : 79.8   Mean   :31.99   Mean   :0.4719   Mean   :33.24            
    ##  3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262   3rd Qu.:41.00            
    ##  Max.   :846.0   Max.   :67.10   Max.   :2.4200   Max.   :81.00

# \<Splitting the dataset\>

Split the dataset. Define an 80:20 train:test split ratio of the dataset
(80% of the original data will be used to train the model and 20% of the
original data will be used to test the model).

``` r
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.8, list = FALSE)
diabetes_dataset_train <- PimaIndiansDiabetes[train_index, ]
diabetes_dataset_test <- PimaIndiansDiabetes[-train_index, ]
```

## \<Classification: SVM with Repeated k-fold Cross Validation\>

SVM Classifier using 5-fold cross validation with 3 reps. We train a
Support Vector Machine (for classification) using “diabetes” variable in
the training dataset based on a repeated 5-fold cross validation train
control with 3 reps.

The repeated k-fold cross-validation method involves repeating the
number of times the dataset is split into k-subsets. The final model
accuracy/RMSE is taken as the mean from the number of repeats

``` r
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

diabetes_dateset_model_svm <- caret::train(diabetes ~ ., data = diabetes_dataset_train,
    trControl = train_control, na.action = na.omit, method = "svmLinearWeights2",
    metric = "Accuracy")
```

## \<Test the trained SVM model using the testing dataset\>

We then proceed to train the model using the testing dataset namely
“diabetes_dataset_test”. We opted to use SVM because we had many
variables and one of the benefits of SVM is that it allows us to find
data that is not regularly distributed. It generally does not suffer
from overfitting and performs well when there is a difference in
classes.

``` r
predictions_svm <- predict(diabetes_dateset_model_svm, diabetes_dataset_test[, 1:9])
```

## \<View a summary of the model and view the confusion matrix\>

From the SVM performing on the PimaIndians dataset we can see that a
pregnant woman is less likely to get diabetes given the different
variables the test dataset had. On a confidence level of 95% we can say
that the model was 72% accurate in predicting the likelihood of diabetes
in pregnant women.

``` r
print(diabetes_dateset_model_svm)
```

    ## L2 Regularized Linear Support Vector Machines with Class Weights 
    ## 
    ## 615 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 492, 492, 492, 492, 492, 492, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cost  Loss  weight  Accuracy   Kappa     
    ##   0.25  L1    1       0.5810298  0.09046601
    ##   0.25  L1    2       0.6411924  0.16633813
    ##   0.25  L1    3       0.5739837  0.10544506
    ##   0.25  L2    1       0.7523035  0.40269746
    ##   0.25  L2    2       0.7349593  0.45644457
    ##   0.25  L2    3       0.4948509  0.14267109
    ##   0.50  L1    1       0.6563686  0.08672517
    ##   0.50  L1    2       0.6281843  0.07256765
    ##   0.50  L1    3       0.6281843  0.11736994
    ##   0.50  L2    1       0.7528455  0.40257178
    ##   0.50  L2    2       0.7252033  0.43271315
    ##   0.50  L2    3       0.4948509  0.14267109
    ##   1.00  L1    1       0.6043360  0.16899489
    ##   1.00  L1    2       0.5902439  0.09349712
    ##   1.00  L1    3       0.5761518  0.11923027
    ##   1.00  L2    1       0.7485095  0.39595285
    ##   1.00  L2    2       0.7230352  0.43088995
    ##   1.00  L2    3       0.4953930  0.14431984
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were cost = 0.5, Loss = L2 and weight = 1.

``` r
caret::confusionMatrix(predictions_svm, diabetes_dataset_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg  88  24
    ##        pos  12  29
    ##                                           
    ##                Accuracy : 0.7647          
    ##                  95% CI : (0.6894, 0.8294)
    ##     No Information Rate : 0.6536          
    ##     P-Value [Acc > NIR] : 0.001988        
    ##                                           
    ##                   Kappa : 0.4512          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.066753        
    ##                                           
    ##             Sensitivity : 0.8800          
    ##             Specificity : 0.5472          
    ##          Pos Pred Value : 0.7857          
    ##          Neg Pred Value : 0.7073          
    ##              Prevalence : 0.6536          
    ##          Detection Rate : 0.5752          
    ##    Detection Prevalence : 0.7320          
    ##       Balanced Accuracy : 0.7136          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

**etc.** as per the lab submission requirements. Be neat and communicate
in a clear and logical manner.
