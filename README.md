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

# Milestone 4:

Creation of few ML models to predict the price of the listing per night and evaluate them.
Use sklearn to compute the key measures of performance for your regression model. That should include the RMSE, and R^2 for both the training and test sets.

Create a function called custom_tune_regression_model_hyperparameters which performs a grid search over a reasonable range of hyperparameter values.
SKLearn has methods like GridSearchCV to do this, BUT instead of using them out the box, use this task to implement a similar thing from scratch to ensure that you have a solid understanding of what's going on under the hood.
The function should take in as arguments:
The model class
The training, validation, and test sets
A dictionary of hyperparameter names mapping to a list of values to be tried
It should return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.
The dictionary of performance metrics should include a key called "validation_RMSE", for the RMSE on the validation set, which is what you should use to select the best model.

Create a function called tune_regression_model_hyperparameters which uses SKLearn's GridSearchCV to perform a grid search over a reasonable range of hyperparameter values.

Create a folder called models.
Within your models folder, create a folder called regression to save your regression models and their metrics in.
Define a function called save_model which saves the model in a file called model.joblib, its hyperparameters in a file called hyperparameters.json, and its performance metrics in a file called metrics.json once it's trained and tuned.
The function should take in the name of a folder where these files should be saved as a keyword argument "folder". In this case, set that argument equal to models/regression/linear_regression.

We have to improve the performance of the model by using different models provided by sklearn.
Use decision trees, random forests, and gradient boosting. Make sure you use the regression versions of each of these models, as many have classification counterparts with similar names.
It's extremely important to apply your tune_regression_model_hyperparameters function to each of these to tune their hyperparameters before evaluating them. Because the sklearn API is the same for every model, this should be as easy as passing your model class into your function.
Save the model, hyperparameters, and metrics in a folder named after the model class. For example, save your best decision tree in a folder called models/regression/decision_tree.
Define all of the code to do this in a function called evaluate_all_models
Call this function inside your if __name__ == "__main__" block.

Define a function called find_best_model which evaluates which model is best, then returns you the loaded model, a dictionary of its hyperparameters, and a dictionary of its performance metrics.
Call this function inside your if __name__ == "__main__" block, just after your call to evaluate_all_models.

# Milestone 5

Start by importing your load_airbnb function defined earlier and using it to load in a dataset with the "Category" as the label.
Use sklearn to train a logistic regression model to predict the category from the tabular data.

Use sklearn to compute the key measures of performance for your classification model. That should include the F1 score, the precision, the recall, and the accuracy for both the training and test sets.

Create a function called tune_classification_model_hyperparameters which does the same thing as the tune_regression_model_hyperparameters function you defined earlier but evaluates the performance using a different metric.
As with the earlier function, it should take in as arguments:
The model class
The training, validation, and test sets
A dictionary of hyperparameter names mapping to a list of values to be tried
And it should return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.
The dictionary of performance metrics should include a key called "validation_accuracy", for the accuracy on the validation set, which is what you should use to select the best model.

Now you need to save the classification model.
Within your models folder, create a folder called classification to save your classification models and their metrics in.
Call the save_model function which you defined earlier to save the classification model, its hyperparameters, and metrics.
In this case, set the folder argument equal to models/classification/logistic_regression.

Improve the performance of the model by using different models provided by sklearn.
Use decision trees, random forests, and gradient boosting. Make sure you use the classification versions of each of these models, as many have regression counterparts with similar names, which you should have used earlier.
It's extremely important to apply your tune_classification_model_hyperparameters function to each of these to tune their hyperparameters before evaluating them. Again, this should be as easy as passing your model class into your function.
Like you did earlier, save the model, hyperparameters, and metrics in a folder named after the model class, but this time in the classification folder within the models folder.
Adapt your evaluate_all_models function which you defined earlier so that it takes in a keyword argument called task_folder. Change your earlier code so that in the earlier regression case, that parameter was set to models/regression. In this case, it should be set to models/classification.

Adapt your find_best_model function defined earlier so that it takes in your task_folder as a parameter and looks inside there to find the models which it should compare.
Like earlier, it should still return you the loaded model, a dictionary of its hyperparameters, and a dictionary of its performance metrics.
Call this function inside your if __name__ == "__main__" block just after your call to evaluate_all_models.

#Milestone 6

Created a PyTorch Dataset called AirbnbNightlyPriceImageDataset that returns a tuple of (features, label) when indexed. The features should be a tensor of the numerical tabular features of the house. The second element is a scalar of the price per night.

Created a dataloader for the train set and test set that shuffles the data. Further, split the train set into train and validation.

Defined a PyTorch model class containing the architecture for a fully connected neural network.
To start with, it should only ingest the numerical tabular data. We will process the text and image features later.
Don't train it yet. Instead, just ensure that it can perform a forward pass on a batch of data and produce an output of the correct shape.
To to this, defined the start of a function called train which takes in the model, the data loader, and the number of epochs. For now, we have to get the first batch of data from the dataloader and pass it through the model, then break out of the training loop.

Completed the training loop so that it iterates through every batch in the dataset for the specified number of epochs, and optimises the model parameters.

Used tensorboard to visualize the training curves of the model and the accuracy both on the training and validation set.

Created a YAML file called nn_config.yaml, next to your modelling.py file, that defines the architecture of the neural network.
Specified:
The name of the optimiser used under a key called optimiser
The learning rate
The width of each hidden layer under a key called hidden_layer_width (For simplicity, make all of the hidden layers the same width)
The depth of the model
Then, defined a function called get_nn_config which reads in this file and returns it as a dictionary.
Passed the config into your train function as the hyperparameter dictionary which you define earlier.
Specified a keyword argument called "config" which must be passed to your model class upon initialisation.
The network should then use that config to set the corresponding hyperparameters.

Created a new folder named neural_networks. Inside it, create a folder called regression.
Adapted the function called save_model so that it detects whether the model is a PyTorch module, and if so, saves the torch model in a file called model.pt, its hyperparameters in a file called hyperparameters.json, and its performance metrics in a file called metrics.json.
Your metrics should include:
The RMSE loss of your model under a key called RMSE_loss for training, validation, and test sets
The R^2 score of your model under a key called R_squared for training, validation, and test sets
The time taken to train the model under a key called training_duration
The average time taken to make a prediction under a key called inference_latency

Defined a function generate_nn_configs which creates many config dictionaries for your network.
Defined a function called find_best_nn which calls this function and then sequentially trains models with each config. It should save the config used in the hyperparameters.json file for each model trained. Return the model, metrics, and hyperparameters. Saved the best model in a folder.


