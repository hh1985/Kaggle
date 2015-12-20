# Load data ----------------------------
require(readr)
require(reshape2)
require(ggplot2)
require(caret)
require(Matrix)
require(xgboost)

setwd("D:/Data/Kaggle/WintonStockMarket")
train <- read_csv("train.csv")
# 4000 x 211

test <- read_csv("test.csv")
# 60000 x 147

# Error function.
wmae <- function(pred, dtrain) {
  target <- getinfo(dtrain, 'label')
  weight <- getinfo(dtrain, 'weight')
  error <- mean(weight * abs(target - pred))
  return(list(metric = "error", value = error))
}

# data profile --------------------------

mtrain <- melt(train, id = c("Id", paste("Feature", 1:25, sep = "_"), "Ret_MinusTwo", "Ret_MinusOne", "Ret_PlusOne", "Ret_PlusTwo", "Weight_Intraday", "Weight_Daily"))
mtrain$variable <- as.numeric(gsub("Ret_", "", mtrain$variable))

ggplot(data = mtrain[mtrain$Id %in% 1:10, ], aes(x = variable, y = value, group = Id)) + geom_line()

cummulative.data <- train[, grep("Ret_[0-9]+", colnames(train))]
for (i in 2:ncol(cummulative.data)) {
  cummulative.data[, i] <-  rowMeans(cummulative.data[, 1:i], na.rm = TRUE)
}
cummulative.data$Id <- train$Id
ctrain <- melt(cummulative.data, id = c("Id"))
ctrain$variable <- as.numeric(gsub("Ret_", "", ctrain$variable))
ggplot(data = ctrain[ctrain$Id == 24, ], aes(x = variable, y = value, group = Id)) + geom_line()

day.return <- train[, c(1, grep("Ret_[PM]", colnames(train)))]
mday.return <- melt(day.return, id = c("Id"))
mday.return$variable <- sapply(mday.return$variable, function(x) {
  if (x == "Ret_MinusTwo") {
    return(-2)
  } else if (x == "Ret_MinusOne") {
    return(-1)
  } else if (x == "Ret_PlusOne") {
    return(1)
  } else if (x == "Ret_PlusTwo") {
    return(2)
  }
})
ggplot(data = mday.return[mday.return$Id %in% 1:20, ], aes(x = variable, y = value, group = Id)) + geom_line()


# Experiment 3 ---------------------------------------------------
day.points <- grep("Ret_[0-9]+", colnames(train))
train$Ret_Zero <- rowMeans(train[, day.points], na.rm = TRUE) * length(day.points)

featurePlot(x = train[, grep("Feature_", colnames(train))],
            y = train$Ret_Zero,
            plot = "box",
            layout = c(5, 5))

day.slice <- train[, c(1, day.points)]

day.slice.median <- apply(train[, day.points], 2, median, na.rm = T)
plot(2:180, day.slice.median)
# Conclusion, the change of median is very small.

stable.curve <- function(d) {
  d2 <- d
  for(i in 3:ncol(d2)) {
    d2[, i] <- rowMeans(d[, 2:i], na.rm = T)
  }
  mdata <- melt(d2, id = c("Id"))
  mdata$variable <- as.numeric(gsub("Ret_", "", mdata$variable))
  g <- ggplot(data = mdata[mdata$Id %in% 1:100, ], aes(x = variable, y = value, group = Id)) + geom_line()
  d2$StableMean <- apply(d2[, 50:120], 1, mean, na.rm = T)
  d2$StableSD <- apply(d2[, 50:120], 1, mean, na.rm = T)
  return(list(d2, g))
}
day.slice.mean <- day.slice
for(i in 3:ncol(day.slice.mean)) {
  day.slice.mean[, i] <- rowMeans(day.slice[, 2:i], na.rm = T)
}
mdata <- melt(day.slice.mean, id = c("Id"))

mdata$variable <- as.numeric(gsub("Ret_", "", mdata$variable))
ggplot(data = mdata[mdata$Id %in% 1:100, ], aes(x = variable, y = value, group = Id)) + geom_line()

day.slice.mean$StableMean <- apply(day.slice.mean[, 50:120], 1, mean, na.rm = T)
day.slice.mean$StableSD <- apply(day.slice.mean[, 50:120], 1, sd, na.rm = T)

# Experiment 4 --------------------------------------------------------------------
train.normal <- train[, 2:28]
train.normal$Response <- day.slice.mean$StableMean * 100
#train.normal$Response <- day.slice.mean$StableSD * 100
#train.normal$Response <- train$Ret_121
train.normal[is.na(train.normal)] <- 0
sparse.matrix <- sparse.model.matrix(Response~., data = train.normal)
dtrain <- xgb.DMatrix(data = sparse.matrix, label = train.normal$Response)
param <- list(objective = "reg:linear", max.depth = 4, eta = 0.1)
bst <- xgboost(params = param, data = dtrain, nthread = 2, nrounds = 200)
xgbImp <- xgb.importance(dimnames(sparse.matrix)[[2]], model = bst)

# Experiment 5 -------------------------------
train.proj <- train[, 29:148] * 100
train.proj$StableMean <- day.slice.mean$StableMean * 100
train.proj$StableSD <- day.slice.mean$StableSD * 100
train.proj[is.na(train.proj)] <- 0
sparse.matrix <- sparse.model.matrix(Ret_121~., data = train.proj)
dtrain <- xgb.DMatrix(data = sparse.matrix, label = train.proj$Ret_121)
param <- list(objective = "reg:linear", max.depth = 4, eta = 0.1)
bst <- xgboost(params = param, data = dtrain, nthread = 2, nrounds = 200)
xgbImp <- xgb.importance(dimnames(sparse.matrix)[[2]], model = bst)
xgb.plot.importance(xgbImp[1:20, ])

# Experiment 6 -------------------------------
plot(train[, "Weight_Intraday"]~day.slice.mean$StableMean)
plot(train[, "Weight_Intraday"]~day.slice.mean$StableSD)

# Weight is associated with stability.
# Weight is something similar to number of stocks.
res.point <- do.call(cbind, lapply(train[, day.points], function(x) {x * train[, "Weight_Intraday"]}))
res.day <- do.call(cbind, lapply(train[, grep("Ret_[PM]", colnames(train))], function(x) {x * train[, "Weight_Daily"]}))

# Mean absolute error.
mae <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(abs(preds - labels))
  return(list(metric = "mae", value = err))
}

res.point.smt <- stable.curve(train)

# It's important to track important stocks.
# The wave of stock is probabily caused by stockes in similar sectors.



# Predicting: projection value, background distribution,
