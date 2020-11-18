## Use this script to check that your answers.csv file 
## has a valid format! Usage:
## Rscript --vanilla lint.R

a <- commandArgs(trailingOnly=TRUE)
filename <- if(length(a)==0) "answers.csv" else a[1]

err <- war <- FALSE

acc <- read.table(filename,header=FALSE,nrows=1)[1,1]
answer <- read.csv(filename,skip=1)

if(acc<0 || 1<acc) {
  cat("error: accuracy must be in [0,1].\n")
  err <- TRUE
} else if(acc<0.5) {
  cat("warning: accuracy is below 0.5!\n")
  war <- TRUE
}

if(nrow(answer)!=965) {
  cat("error: wrong number of rows.\n")
  err <- TRUE
}
if(ncol(answer)!=2) {
  cat("error: wrong number of columns.\n")
  err <- TRUE
} else {
  if(colnames(answer)[1]!="class4" || colnames(answer)[2]!="p") {
    cat("error: wrong column names.\n")
    err <- TRUE
  }
  if(!all(answer[,1] %in% c("nonevent","Ia","Ib","II"))) {
    cat("error: wrong class4 (only nonevent, Ia, Ib, II allowed).\n")
    err <- TRUE
  }
  if(!all(0<=answer[,2]) || !(all(answer[,2]<=1))) {
    cat("error: all of p not in [0,1].\n")
    err <- TRUE
  }
}

if(!err) {
  if(!war) {
    cat("Everything seems to be ok.\n")
  } else {
    cat("The file seems to be valid, but there were warnings.\n")
  }
}

