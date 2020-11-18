#!/usr/bin/env Rscript

a <- commandArgs(trailingOnly=TRUE)

filename <- if(length(a)==0) "dummy.csv" else a[1]

## read the true labels
npf <- read.csv("npf_test.csv")

## create class2
npf$class2 <- factor("event",levels=c("nonevent","event"))
npf$class2[npf$class4=="nonevent"] <- "nonevent"

## read the answer file

## 1st line estimate of accuracy
acc <- read.table(filename,header=FALSE,nrows=1)[1,1]

answer <- read.csv(filename,skip=1)
answer$class4 <- factor(answer$class4,levels=levels(npf$class4))
answer$class2 <- factor("event",levels=c("nonevent","event"))
answer$class2[answer$class4=="nonevent"] <- "nonevent"

## Compute the metrics
binaryaccuracy <- mean(answer$class2==npf$class2)
accuracyofaccuracy <- abs(acc-binaryaccuracy)
perplexity <- exp(-mean(log(ifelse(npf$class2=="event",
                                   answer$p,
                                   1-answer$p))))
multiaccuracy <- mean(answer$class4==npf$class4)

cat(sprintf("%f,%f,%f,%f\n",
            binaryaccuracy,
            accuracyofaccuracy,
            perplexity,
            multiaccuracy))

