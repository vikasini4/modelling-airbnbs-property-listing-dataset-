# Modelling-airbnbs-property-listing-dataset-
This project will focus on building a framework to train, tune and evaluate models on several tasks by using AirBnB's property listing dataset.

# Milestone 3:

The first step before evaluating models is to clean the datasets and for that we have to write a code which will load in the tabular dataset.

First of all, create a file which will contain the code you write for this task. The rating columns are lacking values if you look in the Airbnb dataset. Create the function remove rows with missing ratings, which eliminates the rows with missing ratings in these columns, to get started. The dataset should be input as a pandas dataframe, and the output should be of the same format. 
Lists of strings are contained in the "Description" column. The function combine description strings, which concatenates the list elements into a single string, must be defined. Unfortunately, pandas interprets the values as strings whose contents are legitimate Python lists rather than lists. Don't develop a parse-the-string-into-a-list solution from scratch; instead, search up how to accomplish it. There are a lot of empty quotes in the lists; they should be deleted.

For certain rows, the "guests," "beds," "bathrooms," and "bedrooms" columns have empty values. Instead of removing them, create a method called set default feature values and change these entries to 1. The dataset should be input as a pandas dataframe, and the output should be of the same format. 
Put all of the processing-related code in a function called clean tabular data that accepts a raw dataframe, executes the processing-related functions in order using the output from the preceding function, and then returns the processed data. 

Do the following within an if __name__ == "__main__" block:

.Load the raw data in using pandas

.Call clean_tabular_data on it

.Save the processed data as clean_tabular_data.csv in the same folder as you found the raw tabular data.
