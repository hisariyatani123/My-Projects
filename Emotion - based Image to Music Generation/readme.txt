#####################################################################################################
#	Bridging Paintings and Music -  Exploring Emotion based Music Generation through Paintings	#
#	Tanisha Hisariya										#
#	220929356											#
#	Msc Artificial Intelligence									#
#####################################################################################################


In this zip file you will find various files lets see what each files are:- 

1. Wandb_logs.csv - It contains the running details in csv file taken from wandb only
2.Results - It contains the training and validation logs of some trials during training our model.

##################################################################################################

Lets talk about all other important things which You need to run the code. All the links are of QMUL one drive link so Queen Mary people can easily access it. I have provided another google drive link in case first link does not work.



1. data_model - Download the dataset from the mentioned link unzip it and save it as name "data_model". 
	link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/EbNnv4y4dRtHnnjaPCQY5v0B0jFbMDy5UWxTiHt7BchAnQ
	link 2:	https://drive.google.com/file/d/1n-uLQskwO5eO3YyNQDZwN5kh0ynj8brI/view?usp=sharing
	It contains the dataset, all datasets(images and audio) are stored under their emotion labels, the .json files are made as part of our program only
	which says the chosen dataset for train, test and validation.
	The structure should be <your working directory>/data_model/happy/<".jpg" file>

2. gen audio - It contains the output of generated audio which we obtained at inference. each number represents its respective model like 1 represents 	output of model 1. Here is the link from where you can obtained those results. Save the file as name "gen audio". Under each file you can also 
	see a ouput.json file.
	The structure should be gen <your working directory>audio/1/<test.wav files>
	link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/EdAk3Df-pwJOg-6ZOJBNlwMBjjHxnFKlFdn5Vcp5ybZnnw
	link 2: https://drive.google.com/file/d/1qZH7kiMaqvwMkWAtBYqQagV-FCwH_m-a/view?usp=sharing

3. models - It contains all the models weights checkpoint you can load them and try to run the inference. you can download them from the given link, save 	all the models as "models" folder at the same location.
	link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/Eb_Cig6YtZFCsoBttRcfLmwBViqmRK9GTY9ZikgRjCb5vA
	link 2: https://drive.google.com/file/d/1aEo1DzL5lI9Ih8b7EkFGTL4zpC7yqaBa/view?usp=sharing
	The structure should be <your working directory>/models/<.pt files>

4. processed_data - It contains the pre processed tensor generated from code during trainingg of last version of Modified muiscGen model. If you are not 	downloading this file then just create a folder name "processed_data"
	link 1:  https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/EblUVUBDynJHlhxIVJ1AOPQBRhQus9oMplObEHN4IYcuLw
	The structure should be <your working directory>/processed_data/<files>

5. wandb.zip - This folder contains all logs of wandb trial. you can unzip it and have a view of all.

6. calc_fad.py - This is the python file for calculating FAD evaluation metrics
	The structure should be <your working directory>/calc_fad.py

7. calc_kl.py - This is the python file for calculating KL evaluation metrics.
	The structure should be <your working directory>/calc_kl.py

8. gen.py - This is the python file for generating the music, contains the inference code.
	The structure should be <your working directory>/gen.py

9. run_mod.py - This code is called while fine tuning the MusicGen mode. Here mod meaning this code is for modified training of MusicGen.
	The structure should be <your working directory>/run_mod.py

10. run.py - This code is for calling the original fine tuning process of musicGen
	The structure shoul be <your working directory>/run.py

11. train_mod.py - This code contains the training script of Modified musicGen version.
	The structure should be <your working directory>/train_mod.py

12. train.py - This code contains the implementation of original Musicgen code.
	The structure should be <your working directory>/train.py


13. main.ipynb - This the main file from where you will run the models and generate the outputs.
	The structure should <your working directory>/main.ipynb

14.evaluation.ipynb - This is the file for you to evaluate the generated audio using the model across various objective metrices.

##################################################################################################################################3

To run the code -

1. You should create two different virtual environments one for main.ipynb and other for evaluation.ipynb to avoid any issues related to requirements

2. Make sure you are in the same path where all folders and those file are present.

2. After that you can directly run the cells inside main.ipynb file and that will start downloading all the necessary import libraries and then you can run all the cells

3. Once you are done with the training part you can move to the evaluation.ipynb part to evaluate your model and give the evaluation metrics result.