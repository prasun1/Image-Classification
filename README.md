# Image-Classification
The objectives of this assignment are the following: 
 Use/implement a feature selection/reduction technique. Some sort of feature selection or dimensionality reduction must be included in your final problem solution. 
 Experiment with various classification models. 
 Think about dealing with imbalanced data.


Develop predictive models that can determine, given an image, which one of 11 classes it is

Traffic congestion seems to be at an all-time high. Machine Learning methods must be developed to help solve traffic problems. In this assignment, you will analyze features extracted from traffic images depicting different objects to determine their type as one of 11 classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced. For example, the training data contains 10375 cars but only 3 bicycles and 0 people. Classes in the test data are similarly distributed.


DATA DESCRIPTION:

- train.dat: Training set (dense matrix, samples/images in lines, features in columns).
- train.labels: Training class labels (integers, one per line).
- test.dat: Test set (dense matrix, samples/images in lines, features in columns).
- format.dat: A sample submission with 5296 entries randomly chosen to be 1-11.
