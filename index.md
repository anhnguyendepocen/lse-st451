# ST451 Bayesian Machine Learning 

### Lent Term 2020

### Instructors

* [Kostas Kalogeropoulos](https://kostaskalog.github.io/webpage/), [email](mailto:k.kalogeropoulos@lse.ac.uk), Department of Statistics.  *Office hours*: Mondays 10:30-12:30, COL 6.10

### Teaching Assistants
* Gianluca Giudice [email](mailto:g.giudice@lse.ac.uk), Department of Statistics
* Phil Chan, [email](mailto:p.chan@lse.ac.uk), Department of Statistics.  


### Course Information

- **Lectures** on Mondays 13:00–15:00 in NAB.2.04 (except week 6 which is NAB.LG.01).
- **Computer Classes** there are 3 groups: 
  1. Mondays 15:00–16:30 in STC.S018 taught by Kostas Kalogeropoulos
  2. Tuesdays 16:00-17:30 in FAW.4.03 taught by Gianluca Giudice
  3. Thursdays 15:00-16:30 in STC.S08 taught by Phil Chan 
  
 There will be **no reading week**, hence teaching will be concluded on week 10. 

| **Week** | **Topic**                            |
|----------|--------------------------------------|
| 1        | [Bayesian Inference Concepts](#week-1-bayesian-inference-concepts) |
| 2        | [Bayesian Linear Regression](#week-2-bayesian-linear-regression)                  |
| 3        | [Bayesian Model Selection](#week-3-bayesian-model-choice)    |
| 4        | [Classification](#week-4-classification)       |
| 5        | [Variational Bayes](#week-5-variational-bayes)                  |                       |
| 6        | [Graphical Models](#week-6-graphical-models) |
| 7        | [Mixture models and Clustering](#week-7-mixture-models-and-clustering) | 
| 8        | [Markov Chain Monte Carlo](#week-8-markov-chain-monte-carlo)|
| 9        | [Sequential Data](#week-9-sequential-data) |
| 10       | [Gaussian Processes](#week-10-gaussian-processes)           |

### Course Description

The course sets up the foundations and covers the basic algorithms covered in probabilistic machine learning. Several techniques that are probabilistic in nature are introduced and standard topics are revisited from a Bayesian viewpoint. The module provides training in state-of-the-art methods that have been applied successfully for several tasks such as natural language processing, image recognition and fraud detection.

The first part of the module covers the basic concepts of Bayesian Inference such as prior and posterior distribution, Bayesian estimation,  model choice and forecasting. These concepts are also illustrated in real world applications modelled via linear models of regression and classification and compared with alternative approaches.

The second part of the module introduces and provides training in further topics of probabilistic machine learning such as Graphical models, mixtures and cluster analysis, Variational approximation, advanced Monte Carlo sampling methods, sequential data and Gaussian processes. All topics are illustrated via real-world examples and are contrasted against non-Bayesian approaches.

### Prerequisites

Basic knowledge in probability and first course in statistics such as ST202 or equivalent Probability Distribution Theory and Inference; basic knowledge of the principles of computer programming is sufficient (e.g. in any of Python, R, Matlab, C, Java). This is desired rather than essential. 

### Reading

Lecture slides will be **sufficient** for exam purposes but for optional further reading you can check the books below. 

 - [C. M. Bishop, Pattern Recognition and Machine Learning, Springer 2006](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
 - [K. Murphy, Machine Learning: A Probabilistic Perspective, MIT Press, 2012](https://ebookcentral.proquest.com/lib/londonschoolecons/detail.action?docID=3339490)
 - D. Barber, Bayesian Reasoning and Machine Learning, Cambridge University Press 2012
 - S. Rogers and M. Girolami, A First Course in Machine Learning, Second Edition, Chapman and Hall/CRC, 2016
 
 Specific sections are recommended on the sections from each week below.

### Software

**Python** will be used throughout the course. You can either bring your laptop to the computer classes or use the computer room's PC. If you are using your laptop install [Anaconda (Python 3.7 version)](https://www.anaconda.com/download/)

### Formative coursework

Problem sets will be assigned **each week**. They will include theoretical exercises as well as computer-based assignments. They will be **marked** and returned with **feedback**. Also the marks will appear on LSE FOR YOU.

**Immportant Notes**
 - Submit your problem set in **Columbia House Box 34** 
 - Write the **number of your class group** in the first page

### Assessment

An **individual** project will be assigned on **week 7** and will be **due Tuesday, May 12th noon**. You will be required to analyse data of your choice using the taught Bayesian Machine Learning techniques and present your findings through a paper-like report.

During summer term the course is assessed by a 2 hour **written exam**.

The final grade will be determined by the above with equal weights (**50-50\%**)

### Schedule

---
#### Week 1. Bayesian Inference Concepts

[Lecture Slides](/LectureSlides/SlidesWeek01.pdf)

*Topics covered in Lecture*: 
 - Machine Learning and Bayesian Inference
 - Bayes Estimators
 - Credible Intervals
 - Bayesian Forecasting
 - Bayesian Inference via Monte Carlo methods
 
*Further Reading (Optional)*:
 - Murphy, Sections 2.1-2.7, 5.2.1, 5.2.2, 6.6.1 and 6.6.2

[Computer Class Notebook](/ComputerClasses/ComputerClass01.ipynb)

[Computer Class Notebook with the code for the activities](/ComputerClasses/ComputerClass01_CodeActivities.ipynb)

*Topics covered in Computer Class*: 
 - Introduction to Python, e.g.working with arrays, basic operation and plotting
 - Pseudo-Random numbers
 - Bayesian Inference (Point and Interval Estimation, Forecasting) with Monte Carlo


[Problem Set 1](/ProblemSets/ProblemSet01_WithSolutions.pdf)


---
#### Week 2. Bayesian Linear Regression

[Lecture Slides](/LectureSlides/SlidesWeek02.pdf)

*Topics covered in Lecture*: 
 - Bayesian Linear Regression
 - Ridge Regression
 - Lasso Regression
 - Predictive Distribution
 
*Topics covered in Class*:
 - Using R an RStudio
 - Basic commands for vectors and matrices in R
 - Data visualisation in R
 - Loading data in R 
 
*Further Reading (Optional)*:
 - Murphy, Sections: 1.7, 5.3.1, 5.3.3 ,5.7.1, 7.5, 7.6.1 and 7.6.2 
 - Bishop, Sections: 1.1, 3.1.1, 3.1.4, 3.3.1 and 3.3.2

[Computer Class Notebook](/ComputerClasses/ComputerClass02.ipynb)

[Computer Class Notebook with the code for the activities](/ComputerClasses/ComputerClass02_CodeActivities.ipynb)

*Topics covered in Computer Class*: 
 - Working with Pandas data frames
 - Working with 'for' loops in Python
 - Fitting linear regression models
 - Polynomial curve fitting
 - Introduction to training and test error concepts
 - Ridge regression

[Problem Set 2 with solutions](/ProblemSets/ProblemSet02_WithSolutions.pdf)

[Code for Exercise 3](/ProblemSets/ProblemSet02_Exercise3.ipynb)

---
#### Week 3. Bayesian Model Choice

[Lecture Slides](/LectureSlides/SlidesWeek03.pdf)

*Topics covered in Lecture*: 
 - Bayesian inference for multiparameter models
 - Occam's razor
 - Lindley's paradox
 - Unit information priors
 - Training and Test error, Cross-Validation
 
*Further Reading (Optional)*:
 - Murphy, Sections: 5.3 and 7.6 
 - Bishop, Sections: 2.3.6, 3.3.1, 3.3.2, 3.4, 3.5.1 and 3.5.2

[Computer Class Notebook](/ComputerClasses/ComputerClass03.ipynb)

[Data for Computer Class](/ComputerClasses/AutomobileBI.csv)

[Computer Class Notebook with the code for the activities](/ComputerClasses/ComputerClass03_CodeActivities.ipynb)


*Topics covered in Computer Class*: 
 - Creating your own function in Python
 - Performing matrix operations
 - Conducting full MLE analysis, with confidence intervals rather than just point estimates for the regression coefficients
 - Fitting Bayesian Linear Regression models and summarising the posterior of the regressions coefficients
 - Calculate the marginal likelihood / model evidence for linear regression models to perform Bayesian model selection

[Problem Set 3](/ProblemSets/ProblemSet03.pdf)

[Problem Set 3 with solutions](/ProblemSets/ProblemSet03_WithSolutions.pdf)

---
#### Week 4. Classification

[Lecture Slides](/LectureSlides/SlidesWeek04.pdf)

*Topics covered in Lecture*: 
 - Discriminative and Generative models
 - Logistic Regression
 - Newton Rapshon Algorithm
 - Bayesian Central Limit Theorem
 - Misclassification rate, ROC curves and Scoring Rules
 
*Further Reading (Optional)*:
 - Bishop, Sections: 4.2 to 4.5
 - Murphy, Sections: 4.2.1 to 4.2.4, 8.1, 8.2, 8.3.1, 8.3.3, 8.3.7 and 8.4.1 to 8.4.4.

[Computer Class Notebook](/ComputerClasses/ComputerClass04.ipynb)

[Computer Class Notebook with the code for the activities](/ComputerClasses/ComputerClass04_CodeActivities.ipynb)

*Topics covered in Computer Class*: 
 - Working with 'while' loops in Python
 - Coding Newton-Rapshon optimisation in Python
 - Finding Maximum Likelihood Estimates of logistic regression coefficients
 - Fitting Bayesian logistic Regression models and summarising the posterior of their coefficients
 - Calculate the model evidence, BIC for model choice under Bayesian logistic regression 
 - Evaluate predictive performance for binary data: missclassification rate, sensitivity, specificity, ROC curves and area under them, log scoring rule
 - Fitting linear discriminant analysis

[Problem Set 4](/ProblemSets/ProblemSet04.pdf)

[Problem Set 4 with solutions](/ProblemSets/ProblemSet04_WithSolutions.pdf)

[Problem Set 4 Code for Computer Exercises](/ProblemSets/ProblemSet04_CodeExercises.ipynb)

[Credit Card Fraud Data](/ProblemSets/CreditCardFraud.csv)

[Default Data](/ProblemSets/Default.csv)

---

#### Week 5. Variational Bayes

[Lecture Slides](/LectureSlides/SlidesWeek05.pdf)

*Topics covered in Lecture*: 
 - Variational Inference
 - Kulback-Leibler (KL) Divergence
 - Entropy Lower Bound (ELBO)
 - Mean Field Approximation
 - Automatic Variational Inference
 - Stochastic Gradient Descent
 
*Further Reading (Optional)*:
 - Bishop, Sections: 10.1 10.3 10.6. 
 - Murphy, Sections: 21.1 21.2 21.3.1 21.5.
 - Kucukelbir A., Tran D., Ranganath R., Gelman A., Blei D.M. (2016) [Automatic Differentiation Variational Inference](http://www.jmlr.org/papers/volume18/16-107/16-107.pdf). 

[Computer Class Notebook](/ComputerClasses/ComputerClass05.ipynb)

[Computer Class R Markdown](/ComputerClasses/ComputerClass05.Rmd)

[Computer Class Notebook with the code for the activity](/ComputerClasses/ComputerClass05_CodeActivity.ipynb)

[Computer Class R Markdown with the code for the activity](/ComputerClasses/ComputerClass05_CodeActivity.Rmd)

[Toy Example Stan file](/ComputerClasses/ToyExample.stan)

[Logistic Regression Stan file](/ComputerClasses/LogisticRegression.stan)

*Topics covered in Computer Class*: 
 - Mean field approximation
 - Automatic Differentiation Variational Inference (ADVI) in RStan

[Problem Set 5](/ProblemSets/ProblemSet05.pdf)

[VIX Data](/ProblemSets/VIX_201518.csv)

[Problem Set 5 with solutions](/ProblemSets/ProblemSet05_WithSolutions.pdf)

[Notebook with the code for Exercise 2](/ProblemSets/ProblemSet05_CodeExercise2.ipynb)

[R Markdown with the code for Exercise 3](/ProblemSets/ProblemSet05_CodeExercise3.Rmd)

---

#### Week 6. Graphical Models

[Lecture Slides](/LectureSlides/SlidesWeek06.pdf)

*Topics covered in Lecture*: 
 - Introduction to Graphical Models
 - Directed Graphs / Bayesian Networks
 - Undirected Graphs / Markov Random Fields
 - Naive Bayes Classifier
 - Text Classification
 - Ising Model
 - Image Processing
 
*Further Reading (Optional)*:
 - Bishop, Sections: 8.1 8.2 8.3 and optionally 8.4. 
 - Murphy, Sections: 10.1 10.2 10.5 19.1 19.2 19.4.1 21.3.2 and optionally 10.3 10.4 19.3 19.5.

[Computer Class Notebook](/ComputerClasses/ComputerClass06.ipynb)

[Image for the Computer Class](/ComputerClasses/bayes.bmp)

*Topics covered in Computer Class*: 
 - Image denoising
 - Ising model
 - Text classification
 - Document-term matrix
 - Naive Bayes Classifier
 - Working with Pipelines in Python
 - Adding progress bars in Python

[Problem Set 6](/ProblemSets/ProblemSet06.pdf)

---

#### Week 7. Mixture Models and Clustering

[Lecture Slides](/LectureSlides/SlidesWeek07.pdf)

*Topics covered in Lecture*: 
 - Data Augmentation Setup
 - Gaussian Mixtures
 - EM Algorithm
 - Connection With K-means
 - Overfitted Mixtures 
 - Latent Dirichet Allocation
 
*Further Reading (Optional)*:
 - Bishop, Sections: 9.1 to 9.4, 10.2.1 and 10.2.2. 
 - Murphy, Sections: 11.1, 11.2, 11.4.1, 11.4.2, 21.6 and 27.3.

[Computer Class Notebook](/ComputerClasses/ComputerClass07.ipynb)

*Topics covered in Computer Class*: 
 - Fitting Gaussian Mixture models using the EM algorithm
 - Obtaining information on soft allocation of individuals
 - Model Choice within the family of Gaussian Mixtures
 - Bayesian approach with overfitted mixtures

[Problem Set 7](/ProblemSets/ProblemSet07.pdf)

---

#### Week 8. Markov Chain Monte Carlo

[Lecture Slides](/LectureSlides/SlidesWeek08.pdf)

*Topics covered in Lecture*: 
 - Hirerarchical / Multi-level / Panel Data models 
 - Bayesian sparse variable selection
 - Markov Chains
 - Metropolis-Hastings Algorithm
 - Gibbs Sampler 
 - Hamiltonian MCMC
 
*Further Reading (Optional)*:
 - Bishop, Sections: 11.1.4, 11.2, 11.3 and 11.5. 
 - Murphy, Sections: 24.2.1-3, 24.3.1-4, 24.4.1 and 24.5.4.

[Computer Class Notebook](/ComputerClasses/ComputerClass08.ipynb)

[Computer Class R Markdown](/ComputerClasses/ComputerClass08.Rmd)

*Data for Computer Class*

[Diabetes Data](/ComputerClasses/diabetes.data.txt)

[X train](/ComputerClasses/X_train.csv) [y train](/ComputerClasses/y_train.csv) [X test](/ComputerClasses/X_test.csv) [y test](/ComputerClasses/y_test.csv)

*Stan files for Computer Class*

[Linear Regression](/ComputerClasses/LinearRegression.stan) [/ComputerClasses/Linear Regression 2](LinearRegression2.stan) [Horseshoe](/ComputerClasses/horseshoe.stan)

*Topics covered in Computer Class*: 
 - Sampling from the posterior using the Gibbs Sampler in Python
 - Presenting the output of a Markov Chain Monte Carlo (MCMC) ouput 
 - Using Hamiltonian MCMC to sample from the posterior using Stan
 - Presenting MCMC output in Stan
 - Bayesian Sparse Linear Regression with the *horseshoe* prior

[Problem Set 8](/ProblemSets/ProblemSet08.pdf)

---
