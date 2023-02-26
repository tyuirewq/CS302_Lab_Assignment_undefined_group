# -*- coding: utf-8 -*-
"""Bayesian_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c0bCGnm43AGx4VFIcMIXMCPla87HNNvK
"""

if (!require("pacman")) install.packages("pacman")

pacman::p_load(pacman, bnlearn, bnclassify)

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("graph")
BiocManager::install("Rgraphviz")

"""# Here we are reading table through read.table function"""

data <- read.table("2020_bn_nb_data.txt", header = TRUE, col.names = c("EC100", "EC160", "IT101","IT161","MA101","PH100","PH160","HS101", "QP"))

data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor)

"""# Here we are using Hill Climbing Algorithm for generating Bayesian Network"""

bn<- hc(data[,-9],score = 'k2')

plot(bn)
bn

"""# Conditional probability table for each course"""

fitted_bn <- bn.fit(bn, data[,-9]) 
fitted_bn$EC100
fitted_bn$EC160
fitted_bn$IT101
fitted_bn$IT161
fitted_bn$MA101
fitted_bn$PH100
fitted_bn$HS101

"""# Conditional probability table for each course presented in dotplot"""

bn.fit.dotplot(fitted_bn$EC100)
bn.fit.dotplot(fitted_bn$EC100)
bn.fit.dotplot(fitted_bn$EC160)
bn.fit.dotplot(fitted_bn$IT101)
bn.fit.dotplot(fitted_bn$IT161)
bn.fit.dotplot(fitted_bn$MA101)
bn.fit.dotplot(fitted_bn$PH100)
bn.fit.dotplot(fitted_bn$HS101)

"""# Possible grade of PH100 based on evidence"""

prediction.PH100 <- data.frame(cpdist(fitted_bn, nodes = c("PH100"), evidence = (EC100 == "DD" & IT101 == "CC" & MA101 == "CD")))

my_table <- table(prediction.PH100)
my_table

"""# we built a Naive Bayes classifier to predict whether a student qualifies for an internship program or not."""

set.seed(101)
accuracy_results <- c()
for (i in 1:20) {
 sample <- sample.int(n = nrow(data), size = floor(.7*nrow(data)), replace = F)
 data.train <-data[sample,]
 data.test<- data[-sample,]
 nb.grades <- nb(class = "QP",dataset= data.train)
 nb.grades<-lp(nb.grades, data.train, smooth=0)
 p<-predict(nb.grades, data.test)
 accuracy <- bnclassify:::accuracy(p, data.test$QP)
 accuracy_results <- c(accuracy_results, accuracy)
}
plot(nb.grades)

"""# Accuracy of training model"""

mean(accuracy_results)
accuracy_results2 <- c()

"""# We evaluated the performance of the Naive Bayes classifier on the testing data using the 'predict' function. We repeated this experiment 20 times to get a sense of the variability in the performance of the classifier."""

for (i in 1:20) {
 tn <- tan_cl("QP", data.train)
 tn <- lp(tn, data.train, smooth = 1)
 p <- predict(tn, data.test)
 accuracy2 <- bnclassify:::accuracy(p, data.test$QP)
 accuracy_results2 <- c(accuracy_results, accuracy2)
}
plot(tn)