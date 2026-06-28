"""
Creates a subset from the train data(pkl)[already present] and saves it as a pkl file.

python create_test_data.py

"""

import pickle

TRAIN_DATA_PATH = './prepared_data_labeled/train_model_inputs.pkl'
TEST_DATA_PATH = './prepared_data_labeled/test_model_inputs.pkl'


# Open the file in read-binary mode and load the data
with open(TRAIN_DATA_PATH, 'rb') as file:
    loaded_data = pickle.load(file)

print(f"Train DATA length: {len(loaded_data)}")


TEST_FRACTION = 0.15
TILL_IDX = int(len(loaded_data) * TEST_FRACTION)

test_data = loaded_data[:TILL_IDX]


with open(TEST_DATA_PATH, 'wb') as file:
    pickle.dump(test_data, file)


print(f"Generated TEST DATA length: {len(test_data)}")
