# Numerical Valuation Methods on Compound Option Pricing
## Introduction
This assignment aims to analyze the pricing methodology of Compound American Option. Considering the problem that Compound Option has no analytical solution,
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
  * derivative_pricing_models.py: contains all the pricing and test function, including the binomial tree, trinomial tree, monte-carlo simulation and etc;
  * binomial_tree_test.py: test the accuracy of binomial tree pricing model with BS formula;
  * greeks_result.py: calculate the greeks;
  * implied_volatility.py: try to find the volatility smile;
  * sensitivity_analysis_on_option_price.py: perform the sensitivity analysis on option pricing models;
  * simulation_and_pricing_result.py: output the result.
* Folder "data": containing all the data required, the price of stock option of JP Morgan;
* Folder "result": containing the output figures.

## Ideas
* Firstly, I established two compound option pricing models: Binomial
tree and Trinomial tree. Starting with stock price changes, then
calculated the price of European options on the basis of stock price changes, and
then constructed the pricing method of American options based on the price of European
options. After pricing the underlying option, further applied the binomial
tree and the trinomial tree model to American Compound Options, and the model
results showed that the pricing effect was good.
* Then, I conducted a detailed analysis of each parameter in the pricing
model, mainly analyzing the relationship between the changes of parameters and
the price of compound options.
* Then on the assumption that the stock price follows a lognormal distribution,
I used Monte Carlo Simulation to simulate the price of European Compound
Options, and used Least Square Monte Carlo to simulate the price of American
Compound Options.
* Then, I discussED the Greeks. The analysis of the Greeks is essentially
the sensitivity analysis of option prices.
* Finally, I used Binary and Newtons method to calculate implied volatility
and construct volatility smile.
