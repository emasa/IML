import scipy.io.arff as scparff
import numpy as np
import math

#1. Read and Store in CaseBase and normalize numerical attributes
"""
Parameter data_name: Name of the dataset (i.e.: "adult", "audiology", "auto", ..., "vowel")

Returned value: Array of length ten, containing the ten cross-validation splits.
Each of the ten splits is represented by a 5-tuple (train_case_base, test_case_base, types_numeric, min_vals, max_vals),
where train_case_base and test_case_base is the actual data in the case base
and types_numeric, min_vals and max_vals are the indices of numeric columns and their corresponding minimum and maximum
values which might be needed in the future.

Currently the test case base is not being normalized, but we could easily change that by uncommenting the corresponding
lines in the code below.
"""
def read_cb(data_name):
    data = [0]*10
    FILE_NAME_START = "datasetsCBR/" + data_name + "/" + data_name + ".fold.00000"
    FILE_NAME_END_TRAIN = ".train.arff"
    FILE_NAME_END_TEST = ".test.arff"

    #get types of features from metadata
    meta_data = scparff.loadarff(FILE_NAME_START+"0"+FILE_NAME_END_TRAIN)[1]
    types = np.asarray(meta_data.types())
    types_numeric = np.where(types == 'numeric')[0]

    for i in range(10):
        #read the data and store in a case base (aka numpy array)
        train_name = FILE_NAME_START + str(i) + FILE_NAME_END_TRAIN
        test_name = FILE_NAME_START + str(i) + FILE_NAME_END_TEST
        #we only take the first part of the tuple since we already processed the meta data
        train_case_base = np.asarray(scparff.loadarff(train_name)[0])
        test_case_base = np.asarray(scparff.loadarff(test_name)[0])

        #normalize the data
        min_vals, max_vals = {},{}
        for j in types_numeric:
            vals = [row[j] for row in train_case_base if not math.isnan(row[j])]
            min_val = min(vals)
            max_val = max(vals)

            for k in range(len(train_case_base)):
                train_case_base[k][j] = (float(train_case_base[k][j]) - min_val) / (max_val - min_val)

            #########################################################################################
            #if the test case base has to be normalized aswell, uncomment the following two lines
            #for k in range(len(test_case_base)):
                #test_case_base[k][j] = (float(test_case_base[k][j]) - min_val) / (max_val - min_val)
            #########################################################################################

            #store min_val and max_val for potential later use
            min_vals[j] = min_val
            max_vals[j] = max_val

        data[i] = (train_case_base, test_case_base, types_numeric, min_vals, max_vals)

    return data


#--------------------------------------------------------------------------------------------------------------#
#2. Case Based Reasoner

#a) Retrieval
    def acbr_retrieval_phase():
        pass


#b) Reuse
    def acbr_reuse_phase():
        pass


#c) Revision
    def acbr_revision_phase():
        pass


#d) Review
    def acbr_review_phase():
        pass


#e) Retention
    def acbr_no_retention_phase():
        pass

    def acbr_always_retention_phase():
        pass

    def acbr_retention_phase_1():
        pass

    def acbr_retention_phase_2():
        pass


#f) Comparison: CBR - ACBR


#--------------------------------------------------------------------------------------------------------------#
#3. Comparison: Weighted ACBR with FS - ACBR


#--------------------------------------------------------------------------------------------------------------#
#TESTING
data_name = "autos"
data = read_cb(data_name)
train_case_base, test_case_base, types_numeric, min_vals, max_vals = data[0]
print train_case_base[0]
