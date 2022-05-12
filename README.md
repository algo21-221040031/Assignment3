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
* Then, I discussed the Greeks. The analysis of the Greeks is essentially
the sensitivity analysis of option prices.
* Finally, I used Binary and Newtons method to calculate implied volatility
and construct volatility smile.

## Conclusion
1. The accuracy of binomial tree can be shown as:
.<div align=center>
 <img src="https://user-images.githubusercontent.com/101002984/167990027-c72a76cb-3731-4e28-b7ee-1eaabc8e96f4.png" />
</div>
2. Sensitivity Analysis:
 * 
 .<div align=center>
  <img src="https://user-images.githubusercontent.com/101002984/167990098-754c87bf-5c16-4566-a106-66b0af9b8f10.png" />
  </div>
 *
  .<div align=center>
   <img src="https://user-images.githubusercontent.com/101002984/167990104-0b035c6c-80a9-4535-9b1e-7189b6fc0fbe.png" />
   </div>
 *
  .<div align=center>
   <img src="https://user-images.githubusercontent.com/101002984/167990106-5b0dc699-98b7-41a2-b72d-7ae5fb518c79.png" />
   </div>
 *
   .<div align=center>
    <img src="https://user-images.githubusercontent.com/101002984/167990108-2306d021-ece6-46be-8f7e-5de58527da2b.png" />
    </div>
 *
   .<div align=center>
    <img src="https://user-images.githubusercontent.com/101002984/167990109-38ca9550-4a24-4351-9df7-b0dd93360bc5.png" />
    </div> 
3. Greeks Calculation:
![delta_result](https://user-images.githubusercontent.com/101002984/167990150-adca22ba-9383-4c35-bd5e-3ad63a972222.png)
![gamma_result](https://user-images.githubusercontent.com/101002984/167990157-7ad2da28-a141-497f-b72d-db0f7100417b.png)
![theta_result](https://user-images.githubusercontent.com/101002984/167990158-199b6aa3-2aca-430c-9c61-fb4321ea4796.png)
4. Volatility Smile:
![volatility_smile](https://user-images.githubusercontent.com/101002984/167990187-f93130bb-583b-47d7-b365-f04d8dfa2845.png)
