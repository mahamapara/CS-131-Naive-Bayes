Tufts University CS-131: Aritificial Intelligence Fall 2021

Naive Bayesian Estimation Project: A RADAR TRACE CLASSIFIER 
-------------------------------------------------------------------
A frequent problem at airports is the collision between aircrafts and birds. You are to solve this problem classifying radar tracks at your disposal into two classes: birds and aircrafts. Using a Naïve Recursive Bayesian classifier, your job is to calculate and report the probability that the object belongs to one of the two classes for each datapoint provided.

For your classification, you are given the following data:

1. The PDF of a specified speed for the two categories of objects as represented in the spec.
2. 10 tracks representing the velocity of the unidentified flying object as it was measured by a military-grade radar (1s sampling frequency for a total length of 300s). If the radar was not able to acquire the target and perform the measurement, the corresponding datapoint will be a NaN value

While testing your application, you must consider that in certain cases the speed of the object alone is not sufficient to make a reasonable determination.

Also, assume that the classifier is conservative when transitioning between classes of objects. A probability of transition  and  should be sufficient. However, feel free to change these values as appropriate.

As initial probabilities for the classes, it is normal practice to start the classification from equally distributed values (for two classes, it would be 0.5 for each class). Expect these values to change as the classifier acquires more information from the signals.

Could you extract an additional feature from the data to improve the classification? If yes, can you modify your original solution to also include this feature in the classifier? Make sure to explain your rationale in the README file.

Execution
--------------------------------------
To run the script through the command line: python A5_Mapara.py 

The result is snippets of a 10 data frames, one each object/track. Each data frame has three columns: P(Bird|Obs), P(Plane|Obs) and Time (0 to 299)

README
-----------------------------------------
Instructions and assumptions are in the README.txt file
