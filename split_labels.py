import os
import csv


TrainFile = 'train_labels.csv'
TestFile = 'test_labels.csv'

""" Split the data into training set and testing set """

def split_label(path):
	split_ratio = 0.8	# for training
	# Count the number of files
	with open(os.path.join(path, 'checkerboard_labels.csv'), 'rt') as f_input:
		csv_input = csv.reader(f_input)
		next(csv_input)     
		num_line = len(list(csv_input))
	i = 0
	f_input = open(os.path.join(path, 'checkerboard_labels.csv'), 'rt')
	csv_input = csv.reader(f_input)
	next(csv_input)     # ignore the header line
	train = []
	test = []
	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
	#xml_df = pd.DataFrame(xml_list, columns=column_name)
	for row in csv_input:
		if (i < num_line * split_ratio):
			# For training set
			train.append(row)
			i = i + 1
		else:
			# For testing set
			test.append(row)
			i = i + 1
	# Write the file of training labels
	#train= pd.DataFrame(train, columns=column_name)
	with open(os.path.join(path, TrainFile), 'w') as f_train:
		f_train = csv.writer(f_train)
		f_train.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
		f_train.writerows(train)
		
	# Write the file of testing labels
	#test= pd.DataFrame(test, columns=column_name)
	with open(os.path.join(path, TestFile), 'w') as f_test:
		f_test = csv.writer(f_test)
		f_test.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
		f_test.writerows(test)

def main():

    label_path = os.path.join(os.getcwd(), 'data')
    split_label(label_path)
    print('Successfully splitted the labels.')
    
main()
