library(carat)
library(caret)
library(terra)
library(randomForest)

#Problem 1: Implement both the random forest and neural network approaches shown 
#in the tutorial. Make a map of the predictions for each model. 
#Generate a confusion matrix for each model prediction. Explain the accuracy of 
#each model, potential areas of bias in predictions, and recommend 
#a single model’s predictions to be used for analysis. Explain your reasoning.
oct <- rast("/cloud/project/activity08/Oct_12.tif")
terra::plot(oct)
plotRGB(oct, r=3, g=2, b=1, scale = 0.7, stretch = "lin")

drStack <- rast(c("/cloud/project/activity08/May_19.tif", 
                  "/cloud/project/activity08/June_10.tif",
                  "/cloud/project/activity08/June_18.tif",
                  "/cloud/project/activity08/Oct_12.tif"))
terra::plot(drStack)
plotRGB(drStack, r=3, g=2, b=1, scale = 0.7, stretch = "lin")

lc <- vect("/cloud/project/activity08/land_pts.shp")
terra::plot(lc, "LCID", add=TRUE, legend=FALSE, 
            col=hcl.colors(3,palette="Harmonic"))
head(values(lc))

#Make sure you randomly generate data
#train model to use raster data 
drStackNN <- ifel(is.na(drStack),
                  -1, drStack)
#Uses spatial data
training_points <- subset(lc, lc$train == "train", drop=FALSE)

train <- extract(drStackNN, training_points)
trainTable <- values(training_points)
trainDF <- na.omit(cbind(y=as.factor(trainTable$LCID), 
                         train))

tc <- trainControl(method = "repeatedcv",
                   number = 10,
                   repeats = 10)

nbands <- 20
#
rf.grid <- expand.grid(mtry = 1:round(sqrt(nbands)))

set.seed(43)
#Methods can take a statistical approach for accuracy
#(algorithmic classification based on thresholds and classifications)
rf_model <- caret::train(x=trainDF[,3:22], 
                         y=as.factor(trainDF[,1]),
                         method = "rf",
                         metric="Accuracy",
                         trainControl = tc,
                         tuneGrid=rf.grid)
rf_model


#use raster read in from data
#declares model and data used for model
rf_prediction <- terra::predict(drStackNN, rf_model)
terra::plot(rf_prediction, col=hcl.colors(3, palette="Harmonic"))

rf_prediction_mask <- mask(rf_prediction, drStack[[1]],
                           maskvalue=NaN)
terra::plot(rf_prediction_mask, col=hcl.colors(3,palette="Harmonic"))

validPoints <- subset(lc, lc$train == "valid", drop=FALSE)
valid_table <- values(validPoints)
valid_rf <- extract(rf_prediction_mask, validPoints)
validDF_rf <- data.frame(y=valid_table$LCID, 
                         rf=valid_rf$class)

rf_errorM <- confusionMatrix(as.factor(validDF_rf$rf), 
                             as.factor(validDF_rf$y))
rf_errorM
colnames(rf_errorM$table) <- c("field", "tree", "path")
rownames(rf_errorM$table) <- c("field", "tree", "path")
rf_errorM$table

# Rename rows and columns for better interpretation (optional)
colnames(nn_errorM$table) <- c("field", "tree", "path")
rownames(nn_errorM$table) <- c("field", "tree", "path")
nn_errorM$table
#look for producer and user's accuracy
#applying models
#focus on confusionMatrix

#find size of pixels:
rf_prediction_mask
#error in tutorial
count_rf <- freq(rf_prediction_mask)
count_rf <- count_rf$count * 0.4*0.4
count_rf


# starting parameters for neural net
nnet.grid <- expand.grid(size = seq(from = 1, to = 5, by = 1), 
                         # number of neurons units in the hidden layer 
                         decay = seq(from = 0.001, to = 0.01, by = 0.001)) 
#nnet.grid <- expand.grid(size = seq(from = 1, to = 20, by = 1), 
#                         decay = seq(from = 0.001, to = 0.01, by = 0.001))
# regularization parameter to avoid over-fitting 
set.seed(18)
nnet_model <- caret::train(x = trainDF[,c(3:22)], y = as.factor(trainDF[,1]),
                           method = "nnet", 
                           metric="Accuracy", 
                           trainControl = tc, 
                           tuneGrid = nnet.grid,
                           trace=FALSE)
nnet_model

# predictions
nn_prediction <- terra::predict(drStackNN, nnet_model)

nn_prediction_mask <- mask(nn_prediction,#raster to mask
                           drStack[[1]], # raster or vector with information about mask extent
                           maskvalues=NaN # value in mask that indicates an area/cell should be excluded
)
# make map
terra::plot(nn_prediction_mask, col= hcl.colors(3, palette = "Harmonic"))

#make confusion matrix of nn predictions here
# Extract neural network predictions at validation points
valid_nn <- extract(nn_prediction_mask, validPoints)

# Create a data frame with actual and predicted values
validDF_nn <- data.frame(y = valid_table$LCID, 
                         nn = valid_nn$class)

# Generate the confusion matrix
nn_errorM <- confusionMatrix(as.factor(validDF_nn$nn), 
                             as.factor(validDF_nn$y))

# Print the confusion matrix
nn_errorM

# Optionally, rename columns and rows for better interpretation
colnames(nn_errorM$table) <- c("field", "tree", "path")
rownames(nn_errorM$table) <- c("field", "tree", "path")
nn_errorM$table


freq(nn_prediction)
freq(rf_prediction)

# field RF area calculation
0.4*0.4*71019

# field NN area calculation
0.4*0.4*71047

#Predictions side-by-side
par(mfrow=c(1,2))
terra::plot(nn_prediction_mask, col= hcl.colors(3, palette = "Harmonic"),
            legend=FALSE, axes=FALSE, main="Neural network", box=FALSE)
legend("bottomleft", c("field","tree","path"),
       fill=hcl.colors(3, palette = "Harmonic") ,bty="n")

terra::plot(rf_prediction_mask, col= hcl.colors(3, palette = "Harmonic"),
            legend=FALSE, axes=FALSE, main="Random forest", box=FALSE)
legend("bottomleft", c("field","tree","path"),
       fill=hcl.colors(3, palette = "Harmonic") ,bty="n")

#Question 2: Calculate the NDVI for each drone flight image. Could you use NDVI 
#as a sole indicator for fields, forests, and paths? Explain your answer.
#calculate NDVI pre from NIR (4th band) and Red (3rd band)
ndvi_drStack<- (drStack[[4]]-drStack[[3]])/(drStack[[4]]+drStack[[3]])
#calculate NDVI post
par(mfrow=c(1,2))
terra::plot(ndvi_drStack)

#Question 3: Slightly change the numbers in the training grid 
#for neural networks to include a lower or higher range for the parameter. 
#Describe how that changed to training outcome and what the performance metrics 
#were for the other values. You may include a plot of your training object 
#to help your interpretation.
#Kfold cross validation
nnet.grid <- expand.grid(size = seq(from = 1, to = 5, by = 1), decay = seq(from = 0.001, to = 0.01, by = 0.001)) 
nnet.grid <- expand.grid(size = seq(from = 1, to = 12, by = 1), decay = seq(from = 0.001, to = 0.01, by = 0.001)) 
nnet.grid <- expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.001, to = 0.015, by = 0.001))
nnet.grid <- expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.0001, to = 0.001, by = 0.001)) 

set.seed(43)
rf_model <- caret::train(x = trainDF[,3:22], 
                         y = as.factor(trainDF[,1]), 
                         method = "rf", 
                         metric="Accuracy", 
                         trainControl = tc, 
                         tuneGrid = rf.grid)
rf_model
