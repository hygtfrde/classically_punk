# Welcome to Classically Punk
***

## Task
Given a set of songs 1000 in total and 100 per genre for 10 genres, create a machine learning model that will predict the genre of a song given its input.

## Description
Implement an ETL pipeline to extract audio feature data from song using librosa. Use raw strings to save the data into a CSV file to ensure total data integrity and no loss. Load the CSV file and remove string quotes. Split the set into target column Y for 'genres' and the remianing numerical columns for X. Flatten the 2-dimensional numpy array features in X into 1-dimensional vectors for training. Train a model using feedforward neural network with a single hidden layer where labels are OneHot encoded.

## Installation
Obtain a copy of the music dataset in the `dataset_link.txt` or here:
https://storage.googleapis.com/qwasar-public/track-ds/classically_punk_music_genres.tar.gz 
Activate a Venv or Conda env, then use Pip to install deps.
Run the Music Processor first to extract data, then run the Main Model.
Alternatively, inspect the Jupyter Notebook and run the cells.
## Usage
```
python music_processor.py
python main_model.py
```

### The Core Team


<span><i>Made at <a href="https://qwasar.io">Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt="Qwasar SV -- Software Engineering School's Logo" src="https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png" width="20px" /></span>
