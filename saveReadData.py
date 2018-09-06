import pickle as pc
import os



path=os.getcwd()+'testing_TF/pickle_data.txt'

data =[[1,1],
       [1,2],
       [1,3]]

with open(path, 'w') as f:
	pc.dump(data,f)

with open(path, 'r') as f:
	r_data=pc.load(f)

print r_data


