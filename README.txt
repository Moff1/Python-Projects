Group 10 Term Project - Max, Abdu, Cole, and Moffatt

DataPreProcessing Jupyter Notebook - Takes in 6 .csv datasets (CH4, HFC, etc.) and processes
them together to create the dataset that we used to create our models (DataSet2.csv). You can run 
this code if you would like to see how it works, but it is not necessary to write the dataset to a .csv
becaus I have already included the premade DataSet2.csv file that is the result of the data preprocessing
in this folder.

TermProject_LinearModel.r and TermProject_QuadraticModel.r - R files that contain our linear and quadratc models
and the corresponding stepwise selectiion graphs. All library statements included at the top of each file.
Both take the processed dataset "DataSet2.csv", which is included in this folder. You will need to set your
working directory in R to this folder in order for the read.csv function to work.

PredictingCO2NN Jupyter Notebook -  Jupyter Notebook file that includes our Neural Net models. This, like the .r files,
runs off of the DaraSet2.csv file (processed data)that is included in the turn in folder.
The autokeras might take a few minutes to run. This autokeras model is our best Neural Net model. 
We also have our forward selection graph, but this was inly able to be built upon our handmade neural net.

TermProject_Tree Jupyter Notebook- Jupyter Notebook file that includes our Regressor Tree models.
This, like the .r files, runs off of the DaraSet2.csv file (processed data)that is included in the turn in folder.

I also included our abstract as a pdf and our completed and updated slides.

If you gave any issues or questions with the code, please email me at mtm71515@uga.edu. Thank you!