*Credits to Federico and Joe

*Change Directory
cd "C:\Users\aniru\Documents\GitHub\IO1-PS-CPT\PS2"

*Importing entryData to calculate configuration Probabilities
import delimited config_type.csv, clear

*Run a simple Multinomial Logit model to estimate conditional Probabilites
mlogit config_type x z_1 z_2 z_3, baseoutcome(7)

*Predict Probabilities
predict p1 p4 p7 p8

gen p2 = 0
gen p3 = 0
gen p5 = 0
gen p6 = 0

*Export the Updated file as a new CSV file
export delimited using entryData_updated.csv, replace
