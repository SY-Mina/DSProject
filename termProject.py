import numpy as np
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# encoding
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVC
# deployment
from sklearn.inspection import permutation_importance


dataset = pd.read_csv('aug_train.csv')


pd.set_option('display.max_columns', None)
print('\n###### Check the dataset of the first five lines')
print(dataset.head(5))
print('\n###### Check the dataset')
print(dataset.info())
print('\n###### Check the information of numerical feature')
print(dataset.describe())
print('\n###### Check the number of the null value')
print(dataset.isnull().sum())
print('\n###### Check the total number of the null value')
print(dataset.isnull().sum().sum())


'''
Suitability Check
'''
categorical = (dataset.dtypes == "object")
categorical_list = list(categorical[categorical].index)

print("\nCategorical variables:")
print(categorical_list)

def bar_plot(variable):
    # get feature
    var = dataset[variable]
    # count number of categorical variable
    varValue = var.value_counts()

    # visualize
    plt.figure(figsize=(12, 6))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable, varValue))

# for c in categorical_list:
#     bar_plot(c)

numerical_int64 = (dataset.dtypes == "int64")
numerical_int64_list = list(numerical_int64[numerical_int64].index)

print("Categorical variables:")
print(numerical_int64_list)

def plot_hist(variable):
    plt.figure(figsize = (12,6))
    plt.hist(dataset[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    # plt.show()

# for n in numerical_int64_list:
#     plot_hist(n)

numerical_float64 = (dataset.dtypes == "float64")
numerical_float64_list = list(numerical_float64[numerical_float64].index)

print("Numerical variables:")
print(numerical_float64_list)

# for n in numerical_float64_list:
#     plot_hist(n)


'''
Cleaning Dirty Data
'''
def missing_values_percentage(feature):
    missing_values_number = dataset[feature].isnull().sum()
    if(missing_values_number > 0):
        print(f'{missing_values_number} Missing values in "{feature}" column.')
        missing_value_percentage = 100 * missing_values_number / len(dataset.index)
        print(f'Percentage: {missing_value_percentage:.2f}% of the "{feature}" feature')
        print("------"*10)

print('\n###### Check the missing value percentage')
print("======"*10)
features = list(dataset.columns)

for c in features:
    missing_values_percentage(c)

def fill_missing_values(feature):
    missing_values_number = dataset[feature].isnull().sum()
    if (missing_values_number > 0):

        missing_value_percentage = 100 * missing_values_number / len(dataset.index)

        if (missing_value_percentage <= 0):
            dataset.dropna(subset = [feature], inplace = True)
        else:
            if(missing_value_percentage <=10):
                dataset.dropna(subset=[feature], inplace=True)
                missing_values_number = dataset[feature].isnull().sum()
            else:
                dataset[feature].fillna(method='ffill', limit=4, inplace=True)
                missing_values_number = dataset[feature].isnull().sum()
            if (missing_values_number > 0):
                dataset[feature].fillna(method='bfill', limit=4, inplace=True)

for c in features:
    fill_missing_values(c)


print('\n###### Check the number of the null value')
print(dataset.isnull().sum())

# Divide numerical, categorical columns
num_cols= ['city_development_index' ,'training_hours']
#cat_cols= dataset.drop(['city_development_index' ,'training_hours', 'target'], axis=1).columns

train_label = dataset

rel_exp_idx, edu_idx, comp_size_idx = [list(dataset).index(col) for col in
                                       ['relevent_experience', 'education_level', 'company_size']]

def label_encode(X):
    X.iloc[:, rel_exp_idx] = X.iloc[:, rel_exp_idx].map({'No relevent experience': 0,
                                                         'Has relevent experience': 1}).astype(int)
    X.iloc[:, edu_idx] = X.iloc[:, edu_idx].map({'Primary School': 0,
                                                 'High School': 1,
                                                 'Graduate': 2,
                                                 'Masters': 3,
                                                 'Phd': 4}).astype(int)
    X.iloc[:, comp_size_idx] = X.iloc[:, comp_size_idx].map({'<10': 0,
                                                             '10/49': 1,
                                                             '50-99': 2,
                                                             '100-500': 3,
                                                             '500-999': 4,
                                                             '1000-4999': 5,
                                                             '5000-9999': 6,
                                                             '10000+': 7}).astype(int)
    X.loc[(X['experience'] == '>20'), 'experience'] = 21
    X.loc[(X['experience'] == '<1'), 'experience'] = 0
    X.loc[(X['last_new_job'] == 'never'), 'last_new_job'] = 0
    X.loc[(X['last_new_job'] == '>4'), 'last_new_job'] = 5
    return X

my_encoder = FunctionTransformer(label_encode)
encoded = my_encoder.fit_transform(dataset)
# No city id is required because the city development index contains information about the city
dataset.drop(['city'], axis=1, inplace=True)
dataset.drop(['enrollee_id'], axis=1, inplace=True)
print(dataset)

# remaining data encoding
cat_cols = ['gender', 'enrolled_university', 'major_discipline', 'company_type']

def encoding(encode):
    encode_data = dataset
    ordinal = dataset

    if encode=="label":
        # LabelEncoder
        labelEncoder = LabelEncoder()
        for col in cat_cols:
            encode_data[col] = labelEncoder.fit_transform(encode_data[col])

    else:
    #OrdinalEncoder
        ordinaryEncoder = OrdinalEncoder()
        for col in cat_cols:
            ordinalCol = encode_data[col].values.reshape(-1,1)
            ordinaryEncoder.fit(ordinalCol)
            encode_data[col] = ordinaryEncoder.transform(ordinalCol)
            #print(ordinal[col])

    return encode_data

# print("<categorical encoding>")
# label_data, ordinal_data = encoding()
# print("Label Encoding")
# print(label_data.head(15))
# print("Ordinary Encoding")
# print(ordinal_data.head(15))


def scaling(encode,scaler):
    if scaler == "standard":
        #StandardScaler
        standard = StandardScaler()
        scaler_data = encode
        scaler_data = standard.fit_transform(scaler_data)
        # Train-Test Split
        scaler_data = pd.DataFrame(scaler_data, columns=['city_development_index', 'gender',
                                                                         'relevent_experience', 'enrolled_university',
                                                                         'education_level', 'major_discipline',
                                                                         'experience', 'company_size', 'company_type',
                                                                         'last_new_job', 'training_hours', 'target'])
    elif scaler == "minmax":
        #MinMaxScaler
        minmax = MinMaxScaler()
        scaler_data = encode
        scaler_data = minmax.fit_transform(scaler_data)
        # Train-Test Split
        scaler_data = pd.DataFrame(scaler_data,
                                         columns=['city_development_index', 'gender', 'relevent_experience',
                                                  'enrolled_university', 'education_level', 'major_discipline',
                                                  'experience', 'company_size', 'company_type', 'last_new_job',
                                                  'training_hours', 'target'])
    elif scaler == "robust":
        #RobustScaler
        robust = RobustScaler()
        #scaler_data = encode
        #scaler_data = robust.fit_transform(scaler_data)
        encode = robust.fit_transform(encode)
        # Train-Test Split
        encode = pd.DataFrame(encode,
                                         columns=['city_development_index', 'gender', 'relevent_experience',
                                                  'enrolled_university', 'education_level', 'major_discipline',
                                                  'experience', 'company_size', 'company_type', 'last_new_job',
                                                  'training_hours', 'target'])
    return encode


###########################
def DecisionTree(X,y):
    dtc = DecisionTreeClassifier()
    #float -> int
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    dtc.fit(X_train, y_train)
    return accuracy_score(y_test, dtc.predict(X_test))

#############################
def random_forest(X, y):
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    # standard_data, minmax_data, robust_data = scaling()

    random_forest_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=22)
    random_forest_model.fit(X_train, y_train)
    y_pred_random_forest = random_forest_model.predict(X_test)
    # cm_random_forest = confusion_matrix(y_pred_random_forest, y_test)
    acc_random_forest = accuracy_score(y_test, y_pred_random_forest)

    return acc_random_forest

###########################
def KNNClassifier(X,y):
    #float -> int
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    # Train the KNN classifier
    knn.fit(X_train, y_train)
    # # Check accuracy of the model on the test data
    # score = knn.score(X_test, y_test)

    return accuracy_score(y_test, knn.predict(X_test))


def algorithmmethod(algorithm, X, y):
    score = 0

    if algorithm == "decision":
        score = DecisionTree(X,y)
    elif algorithm == "random":
        score = random_forest(X,y)
    elif algorithm == "knn":
        score = KNNClassifier(X,y)

    return score


#####################################################
def cal_score(encode, scaler, algorithm):
    encode = encoding(encode)
    #print("encode:",encode)
    scaler = scaling(encode,scaler)
    #print("scler:",scaler)
    X = scaler.drop(labels="target", axis=1)
    y = scaler["target"]
    score = algorithmmethod(algorithm, X, y)
    #print("score:",score)
    return score


####################### <open source> ##############################
def openSource(encode,scaler,algorithm):
    #
    labelEncoder = LabelEncoder()
    ordinalEncoder = OrdinalEncoder()
    #
    standard = StandardScaler()
    minmax = MinMaxScaler()
    robust = RobustScaler()
    #
    decision = DecisionTreeClassifier()
    random = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=22)
    knn = KNeighborsClassifier(n_neighbors=3)
    #

    compare = []
    encode_list = []
    scaler_list = []
    algorithm_list = []
    for encode2 in encode:
        for scaler2 in scaler:
            for algorithm2 in algorithm:
                score = cal_score(encode2,scaler2,algorithm2)
                compare.append(score)
                algorithm_list.append(algorithm2)
                encode_list.append(encode2)
                scaler_list.append(scaler2)

    ############# best combination #############
    max = 0
    index_value = 0
    for score in compare:
        if max < score:
            max = score
            index_value = compare.index(max)

    print("best combination:", encode_list[index_value], "+", scaler_list[index_value], "+", algorithm_list[index_value])
    print("accurcy_score_max:", max)

    return encode_list[index_value], scaler_list[index_value], algorithm_list[index_value]
###################################################################


############################ <Main> #################################
encode = ['label', 'ordinal']
scaler = ['standard', 'minmax', 'robust']
algorithm = ['decision', 'random', 'knn']

result_en, result_sc, result_al=openSource(encode,scaler,algorithm)

#############################################################################




# Grid Search- Evaluation (Hyperparameter Tunning)
def Grid_Search(X_train, X_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Grid Search test score: ", grid_search.score(X_test, y_test))
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)


# Make ROC Curve- Evaluation
def roc_curve_plot(y_test, pred_proba_c1):

    # FPRS: False positive rates, TPRS: True positive rates
    # https://velog.io/@sset2323/03-05.-ROC-%EA%B3%A1%EC%84%A0%EA%B3%BC-AUC
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    # Plot ROC Curve as plot curve
    plt.plot(fprs, tprs, label='ROC')
    # Draw center diagonal line
    # The closer to the center line, the less performance
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

# 5-Fold Cross Validation
# K 값도.. 최적화해서 만들어야할까..?
def CrossVal(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    #print(scores)
    print("Cross Validation avg: ", np.mean(scores))


def StratKFold(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    print(scores)
    print("Stratified k-fold Cross Validation avg: ", np.mean(scores))


# Grid Search- Evaluation (Hyperparameter Tunning)
def Grid_Search(model, X_train, X_test, y_train, y_test):
    rf_param_grid = {
        'n_estimators': [10, 30, 50, 70, 100],
        'max_depth': [6, 8, 10, 12],
        'min_samples_leaf': [3, 5, 7, 10],
        'min_samples_split': [2, 3, 5, 10]
    }
    grid_search = GridSearchCV(model, rf_param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Grid Search test score: ", grid_search.score(X_test, y_test))
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

def evaluation(x_test, x_train, y_test, y_train, y_pred, model):
    print("Evaluation of algorithm")
    print(classification_report(y_test, y_pred))
    print("ROC AUC: ", roc_auc_score(y_test, y_pred))
    roc_curve_plot(y_test, model.predict_proba(x_test)[:, 1])

    CrossVal(model, x_test, y_test)
    StratKFold(model, x_test, y_test)

    Grid_Search(model, x_train, x_test, y_train, y_test)

def importance_feature(ls_random, x_train, y_train):
    result = permutation_importance(ls_random, x_train, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(ls_random.feature_importances_)
    tree_indices = np.arange(0, len(ls_random.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,
             ls_random.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(dataset.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(ls_random.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=dataset.columns[perm_sorted_idx])
    fig.tight_layout()
    plt.show()


result_encode = encoding(result_en)
scaled_data = scaling(result_encode, result_sc)

y = scaled_data["target"]
X = scaled_data.drop(labels="target", axis=1)

result_algorithm = result_al

if result_algorithm == "decision":
    model = DecisionTreeClassifier()
    # float -> int
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    model.fit(X_train, y_train)
    pred_model = model.predict(X_test)
    accuracy_score(y_test, model.predict(X_test))
elif result_algorithm == "random":
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=22)
    model.fit(X_train, y_train)
    pred_model = model.predict(X_test)
    cm_random_forest = confusion_matrix(pred_model, y_test)
    acc_random_forest = accuracy_score(y_test, pred_model)
    # print("Random Forest Confusion Matrix: ", cm_random_forest)
    # print("Random Forest ACC: ", acc_random_forest)
elif result_algorithm == "knn":
    #float -> int
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    # Create a KNN classifier
    model = KNeighborsClassifier(n_neighbors=3)
    # Train the KNN classifier
    model.fit(X_train, y_train)
    # # Check accuracy of the model on the test data
    # score = knn.score(X_test, y_test)
    pred_model = model.predict(X_test)
    accuracy_score(y_test, model.predict(X_test))


importance_feature(model, X_train, y_train)
evaluation(X_test, X_train, y_test, y_train, pred_model, model)

# random_forest_model = RandomForestClassifier(max_depth=7, random_state=1)
# random_forest_model.fit(X_train, y_train)
# y_pred_random_forest = random_forest_model.predict(X_test)
# cm_random_forest = confusion_matrix(y_pred_random_forest, y_test)
# acc_random_forest = accuracy_score(y_test, y_pred_random_forest)
#
# importance_feature(random_forest_model, X_train, y_train)
# evaluation(X_test, X_train, y_test, y_train, y_pred_random_forest, random_forest_model)


# score_KNN = KNNClassifier(X,y)
# print(score_KNN)
#
# random_forest(X, y)
#
# score_dicision = DecisionTree(X,y)
# print(score_dicision)
