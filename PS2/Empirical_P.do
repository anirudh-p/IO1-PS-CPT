*Importing entryData to calculate configuration Probabilities
import delimited entryData.csv, clear

*Run a simple Multinomial Logit model to estimate conditional Probabilites
mlogit y X Z_1 Z_2 Z_3, baseoutcome(7)

*Predict Probabilities
predict p1 p2 p3 p5

gen p4 = 0
gen p6 = 0
gen p7 = 0
gen p8 = 0
