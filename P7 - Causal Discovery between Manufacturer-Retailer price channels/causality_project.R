# Load the libraries

# To install pcalg library you may first need to execute the following commands:
source("https://bioconductor.org/biocLite.R")
biocLite("graph")
biocLite("RBGL")
biocLite("Rgraphviz")

require("vars")
require("urca")
require("pcalg")



# Read the input data
data <- read.csv("data.csv")

# Build a VAR model
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
optimalLags <- VARselect(data, lag.max = 10)
varModel <- VAR(data, p = optimalLags$selection["SC(n)"])

# Extract the residuals from the VAR model
# see ?residuals
residuals <- residuals(varModel)

# Check for stationarity using the Augmented Dickey-Fuller test
# see ?ur.df
adfTest <- apply(residuals, 2, ur.df)
lapply(adfTest, summary)

# Check whether the variables follow a Gaussian distribution
# see ?ks.test
ksTest <- apply(residuals, 2, ks.test, y = "pnorm")

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(x = residuals,
          file = "residuals.csv",
          row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM

# PC Algorithm
suffStat <- list(C = cor(residuals), n = nrow(residuals))
pc_fit <-
  pc(
    suffStat,
    indepTest = gaussCItest,
    alpha = 0.1,
    labels = colnames(residuals),
    verbose = TRUE
  )
plot(pc_fit, main = "PC Output")

# LiNGAM Algorithm
lingam(residuals, verbose = TRUE)
