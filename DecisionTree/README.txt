This program builds a decision tree using entropy, majority error, or gini index to find the information gain

To run this program you need to run the shell script with the following format
"run.sh <train_data_path>, <test_data_path>, <max_depth>, <func_id>, <columns>, <label>
where

train_data_path is a string containing the direct path to the training data csv file
test_data_path is a string containing the direct path to the test data csv file
max_depth is an integer that represents the maximum depth of the tree to be created
func_id is either 0, 1, or 2 where
	0 represents using entropy for information gain
	1 represents using majority error
	2 represents using gini index
columns is a list containing the column names of the training and testing data
label is a string containing the column name of the label