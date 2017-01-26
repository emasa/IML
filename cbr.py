import scipy.io.arff as scparff
import numpy as np
import math
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import mutual_info_classif
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
        data[i] = (train_case_base, test_case_base)

    return meta_data, data


def normalize(data, meta_data):
    extra_meta_data = meta_info(meta_data)

    # normalize the data
    train = data[0][0]
    test = data[0][1]
    case_base = np.vstack([train, test])

    types = np.asarray(meta_data.types())[:-1]
    types_numeric = np.where(types == 'numeric')[0]

    min_vals, max_vals = {}, {}
    for j in types_numeric:
        vals = [case[0][j] for case in case_base if not math.isnan(case[0][j])]
        if vals:
            min_val = min(vals)
            max_val = max(vals)
        else:
            min_val = 0
            max_val = 1

        for (train_case_base, test_case_base) in data:
            for k in range(len(train_case_base)):
                if not math.isnan(train_case_base[k][0][j]):
                    train_case_base[k][0][j] = (float(train_case_base[k][0][j]) - min_val) / (max_val - min_val)
                else:
                    train_case_base[k][0][j] = (max_val + min_val)/2

            for k in range(len(test_case_base)):
                if not math.isnan(test_case_base[k][0][j]):
                    test_case_base[k][0][j] = (float(test_case_base[k][0][j]) - min_val) / (max_val - min_val)
                else:
                    test_case_base[k][0][j] = (max_val + min_val)/2

        # store min_val and max_val for later use
        min_vals[j] = min_val
        max_vals[j] = max_val

    types_nominal = np.where(types == 'nominal')[0]
    for j in types_nominal:
        categories = extra_meta_data[j][2]
        for (train_case_base, test_case_base) in data:
            for k in range(len(train_case_base)):
                key = train_case_base[k][0][j]
                if key != "?":
                    train_case_base[k][0][j] = categories[key]
                else:
                    train_case_base[k][0][j] = -1

            for k in range(len(test_case_base)):
                key = test_case_base[k][0][j]
                if key != "?":
                    test_case_base[k][0][j] = categories[key]
                else:
                    test_case_base[k][0][j] = -1


#for future generations
#def normalize(self, case):
    #for j in self.types_numeric:
        #case[j] = (float(case[j]) - self.min_vals[j]) / (self.max_vals[j] - self.min_vals[j])
    #return case


#----------------------------------------------------------------------------------------------------------------------#
#2. Case Based Reasoner

class caseBasedReasoner:

    def __init__(self, train_case_base, meta_data, weights):
        self.case_memory = np.asarray([[case[0], case[1], [0.5], [0.5]] for case in train_case_base])
        self.case_base = train_case_base
        self.meta_data = meta_data
        self.types = np.asarray(self.meta_data.types())[:-1]
        self.types_numeric = np.where(self.types == 'numeric')[0]
        self.weights = weights

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
    def acbr_retrieval_phase(self, current_instance, k):
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

        if use_weighting:
            candidates = self.weighted_acbr_retrieval_phase(new_case[0], k)
        else:
            candidates = self.acbr_retrieval_phase(new_case[0], k)

        prediction = self.acbr_reuse_phase(new_case[0], candidates)
        self.acbr_revision_phase(new_case, prediction, storage)

        if use_oblivion:
            self.acbr_review_phase(new_case, candidates, 0.2)

        if retention_strategy == "DS":
            self.acbr_always_retention_phase(new_case)
        elif retention_strategy == "DD":
            self.acbr_dd_retention_phase(new_case, candidates)
        elif retention_strategy == "DE":
            self.acbr_de_retention_phase(new_case, candidates, prediction)
        else:
            self.acbr_no_retention_phase(new_case)

    # --------------------------------------------------------------------------------------------------------------#
    # 3. Weighted ACBR with FS

    def weighted_euclidian_distance(self, X1, X2):

        if len(self.types) == len(self.types_numeric):
            x, y, w = np.asarray(X1), np.asarray(X2), np.asarray(self.weights)
            return np.linalg.norm(np.sqrt(w) * (x - y))

        else:
            dist = 0.0
            for i in range(len(self.types)):
                dist += self.weights[i] * self.column_distance(X1[i], X2[i], self.types[i])**2
            return math.sqrt(dist)

    def weighted_acbr_retrieval_phase(self, current_instance, k):
        #compute euclidian distance for each case
        dists = np.asarray([self.weighted_euclidian_distance(current_instance, case[0]) for case in self.case_memory])
        #return the k best cases
        return self.case_memory[np.argpartition(dists, k)[:k]]

    # --------------------------------------------------------------------------------------------------------------#

    def test_cycle(self, test_case_base, cycle_type, k, use_weighting):
        storage = []

        start_time = time.time()

        for case in test_case_base:
            self.acbr_cycle(case, storage, cycle_type, k, use_weighting)

        end_time = time.time()

        accuracy = 1.0 * len(filter(lambda x: x == True, storage)) / len(storage)
        efficiency = end_time - start_time
        case_base_size = len(self.case_memory)

        return accuracy, efficiency, case_base_size

#----------------------------------------------------------------------------------------------------------------------#

def meta_info(arff_meta):
    # auxiliary function to parse (feature name, feature type[, range of values (for categorical data)])
    meta_str = str(arff_meta)
    parsed_lines = [l.strip().split("'s type is ") for l in meta_str.split('\n') if 'type' in l]
    cleaned_lines = [(name, type_range.split(', range is')) for name, type_range in parsed_lines]
    final_lines = [(name, t_r[0]) if len(t_r) == 1 else  (name, t_r[0], t_r[1].translate(None, "(){}' ").split(','))
                   for name, t_r in cleaned_lines]
    final_mapping = [n_t_r if len(n_t_r) == 2 else (n_t_r[0], n_t_r[1], {val: k for k, val in enumerate(n_t_r[2])})
                     for n_t_r in final_lines]

    return final_mapping


def get_weights(data, meta_data):
    train = data[0][0]
    test = data[0][1]
    case_base = np.vstack([train, test])

    extra_meta_data = meta_info(meta_data)
    target_map = extra_meta_data[-1][2]

    X = np.asarray([case[0] for case in case_base])
    Y = np.asarray([target_map[case[1][0]] for case in case_base])
    names = meta_data.names()

    rfc = RandomForestClassifier(n_estimators=20, criterion='gini', min_samples_split=2, random_state=0)
    rfc.fit(X, Y, sample_weight=None)
    rfc_feat_weights = rfc.feature_importances_

    abc = AdaBoostClassifier(base_estimator=None, n_estimators=10, learning_rate=0.6, algorithm='SAMME.R',
                             random_state=0)
    abc.fit(X, Y, sample_weight=None)
    abc_feat_weights = abc.feature_importances_

    # specify which features are discrete (nominal)
    discrete_mask = list(np.asarray(meta_data.types()[:-1]) == 'nominal')
    mutual_info_feat_weights = mutual_info_classif(X, Y, discrete_features=discrete_mask)
    # normalized weights between [0, 1]
    mutual_info_feat_weights = mutual_info_feat_weights / np.sum(mutual_info_feat_weights)

    return rfc_feat_weights, abc_feat_weights, mutual_info_feat_weights

#----------------------------------------------------------------------------------------------------------------------#
#testing function

def test(data_name, cycle_type='NR', k=3, weighting=None):
    meta, data = read_cb(data_name)
    weights = [1]*len(meta.types())
    use_weighting = False
    normalize(data, meta)

    if weighting:
        use_weighting = True
        computed_weights = get_weights(data, meta)
        if weighting == "rf":
            weights = computed_weights[0]

        if weighting == "adaboost":
            weights = computed_weights[1]

        if weighting == "infogain":
            weights = computed_weights[2]

    results = []

    for split in data:
        train_case_base, test_case_base = split
        cbr = caseBasedReasoner(train_case_base, meta, weights)
        results.append(cbr.test_cycle(test_case_base, cycle_type, k, use_weighting))

    results = np.asarray(results)
    return results.mean(0)


#----------------------------------------------------------------------------------------------------------------------#
#UNIT TESTS

#READING
# data_name = "credit-a"
# meta, data = read_cb(data_name)
# normalize(data, meta)
# rfc_feat_weights, abc_feat_weights, mutual_info_feat_weights = get_weights(data, meta)
#
# for p in sorted(enumerate(rfc_feat_weights), key=lambda x: x[1], reverse=True):
#     print(p)
# print "-----------------------------------"
#
# for p in sorted(enumerate(abc_feat_weights), key=lambda x: x[1], reverse=True):
#     print(p)
# print "-----------------------------------"
#
# for p in sorted(enumerate(mutual_info_feat_weights), key=lambda x: x[1], reverse=True):
#     print(p)
# print "-----------------------------------"

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
#print test("autos", "NR")
#print cbr.case_memory[-1]


#----------------------------------------------------------------------------------------------------------------------#
#COMPARISONS

def store_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def read_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def compute_n_k_rs(results):
    n = len(results)
    print n
    k = len(results[0])
    print k
    rs = [1.0 * sum([result[i][0] for result in results]) / n for i in range(k)]
    print rs
    return n, k, rs

def friedman_test(results):
    n, k, rs = compute_n_k_rs(results)
    sum = reduce(lambda x,y: x+y**2, rs, [])
    xi = (sum-(1.0*k*(k+1)**2)/4) * 12*n / (k*(k-1))
    return ((n-1)*xi) / (n*(k-1)-xi)

def nemenyi_test(results, critical_level):
    n, k, rs = compute_n_k_rs(results)
    return critical_level * math.sqrt(k*(k+1)/(6*n))

datasets = ["credit-a", "satimage", "nursery"]
#datasets = ["autos", "autos", "autos"]

# ## k ##
results = []
ks = [1, 3, 5, 7]
for dataset in datasets:
    result = []
    for k in ks:
        result.append(test(dataset, k=k))
        print "finished a k"
    results.append(result)
    print "finished a dataset"
store_to_file(results, "experiments_k")

#results = read_from_file("experiments_k")
#n,k,rs = compute_n_k_rs(results)

#
# ## cycle types ##
# results = []
# cycle_types = ["NR", "DS", "DD", "DE", "DD-O", "DE-O"]
# for dataset in datasets:
#     result = []
#     for type in cycle_types:
#         result.append(test(dataset, cycle_type=type))
#     results.append(result)
# store_to_file(results, "experiments_cycle_types")
#
# ## weighted acbr ##
# results = []
# for dataset in datasets:
#     result = []
#     #todo: change cycle type below
#     result.append(test(dataset, cycle_type=""))
#     result.append(test(dataset, cycle_type="", weighting="rf"))
#     result.append(test(dataset, cycle_type="", weighting="adaboost"))
#     results.append(result)
# store_to_file(results, "experiments_weighting")




#test = [[0, 1, 2], ['a', 'b', 'c']]
#store_to_file(test, "test")
#print read_from_file("test")

