# Term Deposit Predictive Model

## About 

A supervised machine learnning predictive model to classify customers who are most likely to purchase a bank's term sheet. 

## Business Need

> I successfully finished up to my rigorous job interview process with Bank of Portugal
as a machine learning researcher. The investment and portfolio department would want
to be able to identify their customers who potentially would subscribe to their term
deposits. As there has been heightened interest of marketing managers to carefully tune
their directed campaigns to the rigorous selection of contacts, the goal of my employer
is to find a model that can predict which future clients who would subscribe to their term
deposit. Having such an effective predictive model can help increase their campaign
efficiency as they would be able to identify customers who would subscribe to their term
deposit and thereby direct their marketing efforts to them. 

> This would help them better manage their resources (e.g human effort, phone calls, time)
The Bank of Portugal, therefore, collected a huge amount of data that includes
customers profiles of those who have to subscribe to term deposits and the ones who
did not subscribe to a term deposit. As their newly employed machine learning
researcher, they want me to come up with a robust predictive model that would help
them identify customers who would or would not subscribe to their term deposit in the
future.

> My main goal as a machine learning researcher is to carry out data exploration, data
cleaning, feature extraction, and developing robust machine learning algorithms that
would aid them in the department. 


## Dataset Descriptions 
> Dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)


Input variables:
### bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
### related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
### social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')





## Exploratory Data Analysis


## Modelling and Predicton 




## Production
- Tablaeu (visualizations)
- Medium Article (project report)
- WebApp with Flask (to intereact with the predicton model)
