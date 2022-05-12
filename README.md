# Assignment 3
## Introduction
This assignment aims to analyze the pricing methodology of Compound American Option. Considering the problem that Compound Option have no analytical solution,
this assignment focuses on the use of tree models: Binomial tree and Trinomial trees
to price Compound American Option.The pricing models shows great effect on the
Compound Option. In order to further test the pricing effect, Monte
Carlo simulation and Least Square Monte Carlo simulation are used to simulate the European
Compound Call Option and the American Compound Call Option under the
premise of assuming that the stock price obeys the lognormal distribution.

## Language Environment
* Python 3.9
* Modules: pandas, numpy, matplotlib.pyplot

## Files Description
* Folder "code": containing all the coding .py files;
  * derivative_pricing_models.py: contains all the pricing and test function, including the binomial tree, trinomial tree, monte-carlo simulation and etc.
* Folder "data": containing all the data required, the price of stock option of JP Morgan;
* Folder "result": containing the output figures.
