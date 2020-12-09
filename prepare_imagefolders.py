import os
from args import args
import pandas as pd
import logging
from nltk.corpus import stopwords
from preprocess import read_ground_truth_files
import shutil
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
	input_path = "/media/darg1/5EDD3D191C555EB5/WikimediaCommons-scraper/data/final"
	output_path = "/media/darg1/5EDD3D191C555EB5/wikimedia_dataset"
	path = os.path.join(input_path, "absconc_data_raw.csv")
	ground_truth_scores = read_ground_truth_files(args.data_dir)
	field = "caption"

	#read the dataset
	df_data = pd.read_csv(path, delimiter='\t')
	df_field = df_data[['word',field]].dropna()
	logging.info("dataset size:" + str(len(df_data)))
	logging.info("field's non-empty size:" + str(len(df_field)))

	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_field_filtered = df_field[~df_field['word'].isin(stop_words)]
	
	#get the image ids(=names)
	df_field_filtered.reset_index(level=0, inplace=True)
	raw_data = df_field_filtered.to_numpy()

	data = []
	label = []
	for row in raw_data:
		data.append(row[0])
		if ground_truth_scores[row[1]] >= 400:
			label.append(1)
		else:
			label.append(0)


	count=0
	for folder in tqdm(os.listdir(input_path)):
		subpath = os.path.join(input_path, folder)

		if os.path.isdir(subpath):

			for imagefile in tqdm(os.listdir(subpath)):
				extension = imagefile.split(".")[1]
				filename = imagefile.split(".")[0]

				try:
					index = data.index(int(filename))
					_from = os.path.join(subpath, imagefile)
					_to = os.path.join(output_path, str(label[index]))
					shutil.copy(_from, _to)
					count+=1
				except:
					pass
	print(str(count) + "files copied")


	
	






