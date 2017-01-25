import scipy.io.arff as scparff
import numpy as np
import math
import time
from collections import Counter

#1. Read and Store in CaseBase and normalize numerical attributes
"""
Parameter data_name: Name of the dataset (i.e.: "adult", "audiology", "auto", ..., "vowel")

Returned value: Tuple (meta, data) where meta is a tuple(types, types_numeric) representing the meta information with
types being an array containing the type of each column and types_numeric being an array containing the indices of all
numeric types (which is useful for normalizing).
Data is an array of length ten, containing information about the ten cross-validation splits.
Each of the ten splits is represented by a 4-tuple (train_case_base, test_case_base, min_vals, max_vals),
where train_case_base and test_case_base is the actual data in the case base,
and min_vals and max_vals are dictonaries for the minimum and maximum values of numeric columns (needed for normalizing).
"""
def read_cb(data_name):
    data = [0]*10
    FILE_NAME_START = "datasetsCBR/" + data_name + "/" + data_name + ".fold.00000"
    FILE_NAME_END_TRAIN = ".train.arff"
    FILE_NAME_END_TEST = ".test.arff"

    #get types of features from metadata
    meta_data = scparff.loadarff(FILE_NAME_START+"0"+FILE_NAME_END_TRAIN)[1]
    types = np.asarray(meta_data.types())[:-1]
    types_numeric = np.where(types == 'numeric')[0]

    for i in range(10):
        ###read the data and store in a case base (aka numpy array)###
        train_name = FILE_NAME_START + str(i) + FILE_NAME_END_TRAIN
        test_name = FILE_NAME_START + str(i) + FILE_NAME_END_TEST
        #we only take the first part of the tuple since we already processed the meta data
        train_case_base = np.asarray(scparff.loadarff(train_name)[0])
        test_case_base = np.asarray(scparff.loadarff(test_name)[0])
        #convert each case from tuple to list
        train_case_base = [list(case) for case in train_case_base]
        test_case_base = [list(case) for case in test_case_base]
        #split each sample into features and class
        train_case_base = [(case[:-1],case[-1:]) for case in train_case_base]
        test_case_base = [(case[:-1],case[-1:]) for case in test_case_base]
        #convert case base from list to numpy array
        train_case_base = np.asarray(train_case_base)
        test_case_base = np.asarray(test_case_base)

        #normalize the data
        min_vals, max_vals = {},{}
        for j in types_numeric:
            vals = [case[0][j] for case in train_case_base if not math.isnan(case[0][j])]
            if vals:
                min_val = min(vals)
                max_val = max(vals)
            else:
                print "oh no"
                min_val = 0
                max_val = 1

            for k in range(len(train_case_base)):
                train_case_base[k][0][j] = (float(train_case_base[k][0][j]) - min_val) / (max_val - min_val)

            #store min_val and max_val for later use
            min_vals[j] = min_val
            max_vals[j] = max_val

        data[i] = (train_case_base, test_case_base, min_vals, max_vals)

    return (types, types_numeric), data


#--------------------------------------------------------------------------------------------------------------#
#2. Case Based Reasoner
class caseBasedReasoner:

    def __init__(self, train_case_base, types, types_numeric, min_vals, max_vals):
        self.case_memory = np.asarray([[case[0], case[1], [0.5], [0.5]] for case in train_case_base])
        #self.case_base = train_case_base
        self.types = types
        self.types_numeric = types_numeric
        self.min_vals = min_vals
        self.max_vals = max_vals


    ##### a) Retrieval #####
    def normalize(self, case):
        for j in self.types_numeric:
            case[j] = (float(case[j]) - self.min_vals[j]) / (self.max_vals[j] - self.min_vals[j])
        return case

    #Computes the distance of two elements x1, x2
    def column_distance(self, x1, x2, type):
        if type=="numeric":
            if math.isnan(x1) or math.isnan(x2):
                return 1
            return x1-x2
        else:
            return 0.0 if x1==x2 else 1.0

    #Computes the Euclidian distance for two arrays X1,X2
    def euclidian_distance(self, X1, X2):
        if len(self.types) == len(self.types_numeric):
            return np.linalg.norm(np.asarray(X1)-np.asarray(X2))
        else:
            dist = 0.0
            for i in range(len(self.types)):
                dist += self.column_distance(X1[i], X2[i], self.types[i])**2
            return math.sqrt(dist)

    #Selects and returns the k most similar cases to current_instance within the train_case_base
    def acbr_retrieval_phase(self, current_instance, k=3):
        current_instance = self.normalize(current_instance)
        #compute euclidian distance for each case
        dists = np.asarray([self.euclidian_distance(current_instance, case[0]) for case in self.case_memory])
        #return the k best cases
        return self.case_memory[np.argpartition(dists, k)[:k]]


    ##### b) Reuse #####
    #Returns the most likely class of current_instance, given a set of similar cases (candidates).
    #Default policy is "majority vote", but "best case" is used when there is no clear majority
    def acbr_reuse_phase(self, current_instance, candidates):
        classes = [candidate[1][0] for candidate in candidates]
        class_counts = Counter(classes).most_common()
        #If there is a clear majority among the candidate classes return the majority class
        if len(class_counts)==1 or class_counts[0][1] > class_counts[1][1]:
            return [class_counts[0][0]]

        #If there is no clear majority do 'best vote', i.e. return class of most similar retrieved case
        else:
            #compute the Euclidian distance of all candidates
            dists = np.asarray([self.euclidian_distance(current_instance, candidate[0]) for candidate in candidates])
            #get the most similar case
            best = candidates[np.argpartition(dists, 1)[0]]
            #return the corresponding class
            return best[1]


    #c) Revision
    def acbr_revision_phase(self, current_instance, prediction, storage):
        true_class = current_instance[1][0]
        storage.append(true_class == prediction[0])

    #d) Review
    def acbr_review_phase(self, new_case, candidates, alpha):
        true_class = new_case[1][0]
        for candidate in candidates:
            goodness = candidate[3][0]

            #update goodness
            r = 1 if candidate[1][0] == true_class else 0
            goodness = goodness + alpha * (r-goodness)
            index = self.case_memory.tolist().index(candidate.tolist())
            candidate[3] = [goodness]
            self.case_memory[index] = candidate

            #oblivion
            goodness_initial = candidate[2][0]
            if goodness < goodness_initial:
                cm_list = self.case_memory.tolist()
                del cm_list[index]
                self.case_memory = np.asarray(cm_list)


    #e) Retention
    def store_case(self, new_case, goodness):
        self.case_memory = np.append(self.case_memory, np.asarray([[new_case[0], new_case[1], [goodness], [goodness]]]), axis=0)

    def acbr_no_retention_phase(self, new_case):
        pass

    def acbr_always_retention_phase(self, new_case):
        self.store_case(new_case, 0.5)

    def acbr_dd_retention_phase(self, new_case, candidates, threshold=0.5):
        k = len(candidates)
        classes = [candidate[1][0] for candidate in candidates]
        c = len(list(set(classes)))
        majority_class = max(set(classes), key=classes.count)
        majority_cases = filter(lambda x: x[1][0]==majority_class, candidates)
        mk = len(majority_cases)

        #the calculation of d is not possible if c==1 since it results in a null division
        #therefore we simply check that beforehand and set d to 0 if that is the case
        if c < 2:
            d = 0
        else:
            d = 1.0*(k-mk) / ((c-1)*mk)

        if d > threshold:
            x = [case[3][0] for case in majority_cases]
            majority_goodness = max(x)
            self.store_case(new_case, majority_goodness)


    #Since we use majority vote as the standard reuse policy, the LE and DE policies are mostly the same
    def acbr_de_retention_phase(self, new_case, candidates, prediction):
        if new_case[1][0] != prediction[0]:
            classes = [candidate[1][0] for candidate in candidates]
            majority_class = max(set(classes), key=classes.count)
            if new_case[1][0] != majority_class:
                self.store_case(new_case, 0.5)


    # f) Comparison: CBR - ACBR
    #todo: compare different values of k
    #todo: compare different values of alpha
    #todo: Compare no retention to always retention
    #todo: Compare different versions of the acbr cycle (i.e. DD-O vs DD, etc.)
    def acbr_cycle(self, new_case, storage, cycle_type, k, use_weighting):

        use_oblivion = cycle_type.endswith("O")
        retention_strategy = cycle_type[:2]

        print retention_strategy
        print use_oblivion

        if use_weighting:
            candidates = self.weighted_acbr_retrieval_phase(new_case[0], k)
        else:
            candidates = self.acbr_retrieval_phase(new_case[0], k)

        prediction = self.acbr_reuse_phase(new_case[0], candidates)
        self.acbr_revision_phase(new_case, prediction, storage)

        if use_oblivion:
            self.acbr_review_phase(new_case, candidates, 1)

        if retention_strategy == "DS":
            self.acbr_always_retention_phase(new_case)
        elif retention_strategy == "DD":
            self.acbr_dd_retention_phase(new_case, candidates)
        elif retention_strategy == "DE":
            self.acbr_de_retention_phase(new_case, candidates, prediction)
        else:
            self.acbr_no_retention_phase(new_case)


    # --------------------------------------------------------------------------------------------------------------#
    # 3. Weighted ACBR with FS - todo
    def weighted_acbr_retrieval_phase(self):
        pass

    # --------------------------------------------------------------------------------------------------------------#

    def test_cycle(self, test_case_base, cycle_type='NR', k=3, use_weighting=False):
        storage = []
        start_time = time.time()
        for case in test_case_base:
            self.acbr_cycle(case, storage, cycle_type, k, use_weighting)
        end_time = time.time()
        accuracy = 1.0 * len(filter(lambda x: x == True, storage)) / len(storage)
        efficiency = end_time - start_time
        case_base_size = len(self.case_memory)

        return accuracy, efficiency, case_base_size


def test(data_name, cycle_type='NR'):
    meta, data = read_cb(data_name)
    types, types_numeric = meta
    results = []
    for split in data:
        train_case_base, test_case_base, min_vals, max_vals = split
        cbr = caseBasedReasoner(train_case_base, types, types_numeric, min_vals, max_vals)
        results.append(cbr.test_cycle(test_case_base, cycle_type))
    results = np.asarray(results)
    return results.mean(0)





##### UNIT TESTS #####

#READING
#data_name = "autos"
#meta, data = read_cb(data_name)
#types, types_numeric = meta
#train_case_base, test_case_base, min_vals, max_vals = data[0]
#case1 = test_case_base[0]
#case2 = test_case_base[1]
#print types
#print types_numeric

#cbr = caseBasedReasoner(train_case_base, types, types_numeric, min_vals, max_vals)

#RETRIEVAL, REUSE
#cands = cbr.acbr_retrieval_phase(case1[0], 3)
#pred = cbr.acbr_reuse_phase(case1[0], cands)

#REVIEW, RETENTION
#train_case_base = [[[0,0,0,0,0],[0]],[[1,1,1,1,1],[1]],[[2,2,2,2,2],[0]]]
#types = ['numeric']*5
#types_numeric = range(5)
#min_vals, max_vals = [],[]
#cbr = caseBasedReasoner(train_case_base, types, types_numeric, min_vals, max_vals)
#new_case = [[1,2,3,4,5],[0]]
#print cbr.case_memory
#prediction = [1]
#cands = [[[0,0,0,0,0],[0],[0.5],[0.5]],[[1,1,1,1,1],[1],[0.5],[0.5]],[[2,2,2,2,2],[0],[0.5],[0.5]]]
#cbr.acbr_review_phase(new_case, cands, 1)
#cbr.acbr_de_retention_phase(new_case, cands, prediction)
#cbr.acbr_dd_retention_phase(new_case, cands, 0)
#print cbr.case_memory

#CYCLES
#storage = []
#cbr.cbr_cycle(case1, storage)
#cbr.acbr_cycle(case2, storage)
#print storage

#test_cycle(), test()
#print cbr.test_cycle(test_case_base, 'acbr')
print test("credit-a", "NR")
#print cbr.case_memory[-1]
