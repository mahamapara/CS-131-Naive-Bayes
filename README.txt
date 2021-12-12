Maha Mapara
CS 131: A5 Naive Bayesian Classification
11/29/21

---
Instructions and additional info:
1. Libraries needed: pandas, math
2. In addition to the README and .py file, the two txt files pdf.txt and data.txt are needed.
3. To run the script through the command line: python A5_Mapara.py 
The result is snippets of a 10 data frames, one each object/track. Each data frame has three columns: P(Bird|Obs), P(Plane|Obs) and Time (0 to 299)

---
Assumptions and rules for Recursive Bayesian Estimation:
1. NaN values in data.txt were replaced with the mean speed value of that object/track.
2. Both txt files are read in as pandas data frames, where each individual line is a row and comma separted values as columns. 
Example: in the case of the data.txt file, the data frame has 10 rows (one for each object/track) and each column is a speed value at a single time point 
(300 columns for 300 time points)
3. The pdf.txt file when graphed ranges from 0 to 400 speed instead of 0 to 200 like in the spec. Based on this and my conversation with BJ, to get the probability
of being a plane or bird based on speed from pdf, the speed value from an object at a single point was doubled to get the probability so that it is the same as 
when getting it from graph 1 in the spec. 
Example: For prob of being bird for object at row 0 at speed 30 (from data.txt), get probability at the '60th' value (index# 59) from pdf, row 0.
4. If speed value has decimal points, it is rounded to the nearest whole number to get probability.
5. Prior is assumed to be 0.5 for both bird and plane at t = 0.
6. P(S|S') when both states are same is 0.9. When states are different, it is 0.1. 
Example: if P(S|S') is P(Plane|Plane) = 0.9 [the probability of current state being plane if it was plane at the last time point]
P(Plane|Bird) = 0.1 [the probability of current state being plane if it was bird at the last time point]

---
Adding an extra feature for classification:

I couldn't think of adding any feature that could improve classificaion. A tweak that could be made but is not an extra feature is to change the P(S|S') value based 
last actual calculated probability of what the state is instead of setting it at 0.9 and 0.1. Perhaps looking at the mean speed and adding that as a feature can be 
an option but it wouldn't improve the classification drastically. Feature transformations or scaling might be better solution but the probabilties overall were very 
definitive for each object so there likely won't be a great benefit to adding a new feature. 
---

I discussed the assignment, implementation ideas, reading in data, structure and design with BJ.

Online resources used:
http://gki.informatik.uni-freiburg.de/teaching/ws0607/advanced/recordings/BayesFiltering.pdf
https://people.csail.mit.edu/mrub/talks/filtering.pdf
https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
https://stackoverflow.com/questions/50701460/python-replace-na-in-column-with-value-from-corresponding-row-value-of-another
https://thispointer.com/pandas-replace-nan-with-mean-or-average-in-dataframe-using-fillna/