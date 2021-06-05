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
#도시 개발 지수에 도시에 대한 정보가 나와있기 때문에 도시 id는 필요없음
dataset.drop(['city'], axis=1, inplace=True)
dataset.drop(['enrollee_id'], axis=1, inplace=True)
print(dataset)

#나머지 데이터들 인코딩
cat_cols = ['gender', 'enrolled_university', 'major_discipline', 'company_type']

def encoding():
    label = dataset
    ordinal = dataset

    #LabelEncoder
    labelEncoder = LabelEncoder()

    for col in cat_cols:
        label[col] = labelEncoder.fit_transform(label[col])

    #OrdinalEncoder
    ordinaryEncoder = OrdinalEncoder()
    for col in cat_cols:
        ordinalCol = ordinal[col].values.reshape(-1,1)
        ordinaryEncoder.fit(ordinalCol)
        ordinal[col] = ordinaryEncoder.transform(ordinalCol)
        print(ordinal[col])

    return label, ordinal

# print("<categorical encoding>")
# label_data, ordinal_data = encoding()
# print("Label Encoding")
# print(label_data.head(15))
# print("Ordinary Encoding")
# print(ordinal_data.head(15))


def scaling(df):
    #StandardScaler
    standard = StandardScaler()
    standard_data = df
    standard_data = standard.fit_transform(standard_data)
    #MinMaxScaler
    minmax = MinMaxScaler()
    minmax_data = df
    minmax_data = minmax.fit_transform(minmax_data)
    #RobustScaler
    robust = RobustScaler()
    robust_data = df
    robust_data = robust.fit_transform(robust_data)
    return standard_data,minmax_data,robust_data


# label_standard_data,label_minmax_data,label_robust_data=scaling(label_data)
# ordinal_standard_data, ordinal_minmax_data, ordinal_robust_data=scaling(ordinal_data)



###########################
def DecisionTree(X,y):
    dtc = DecisionTreeClassifier()
    #float -> int
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    dtc.fit(X_train,y_train)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(classification_report(y_test, dtc.predict(X_test)))
    return accuracy_score(y_test, dtc.predict(X_test))
    #print('The accuracy score with using the decision tree classifier is :',accuracy_score(y_test, dtc.predict(X_test)))
#############################
def random_forest(X, y):
    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)
    # standard_data, minmax_data, robust_data = scaling()

    random_forest_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=22)
    random_forest_model.fit(X_train, y_train)
    y_pred_random_forest = random_forest_model.predict(X_test)
    cm_random_forest = confusion_matrix(y_pred_random_forest, y_test)
    acc_random_forest = accuracy_score(y_test, y_pred_random_forest)
    # print("Random Forest Confusion Matrix: ", cm_random_forest)
    # print("Random Forest ACC: ", acc_random_forest)
    return acc_random_forest
############################
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
    #print('The accuracy score with using the decision tree classifier is :',accuracy_score(y_test, dtc.predict(X_test)))

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
def Grid_Search(X_train, X_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Grid Search test score: ", grid_search.score(X_test, y_test))
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

def open():
    ##
    print("<categorical encoding>")
    label_data, ordinal_data = encoding()
    print("Label Encoding")
    print(label_data.head(15))
    print("Ordinary Encoding")
    print(ordinal_data.head(15))
    ##
    label_standard_data, label_minmax_data, label_robust_data = scaling(label_data)
    ordinal_standard_data, ordinal_minmax_data, ordinal_robust_data = scaling(ordinal_data)
    ##
    # Train-Test Split
    label_standard_data = pd.DataFrame(label_standard_data, columns=['city_development_index', 'gender',
                                                                     'relevent_experience', 'enrolled_university',
                                                                     'education_level', 'major_discipline',
                                                                     'experience', 'company_size', 'company_type',
                                                                     'last_new_job', 'training_hours', 'target'])
    label_minmax_data = pd.DataFrame(label_minmax_data,
                                     columns=['city_development_index', 'gender', 'relevent_experience',
                                              'enrolled_university', 'education_level', 'major_discipline',
                                              'experience', 'company_size', 'company_type', 'last_new_job',
                                              'training_hours', 'target'])
    label_robust_data = pd.DataFrame(label_robust_data,
                                     columns=['city_development_index', 'gender', 'relevent_experience',
                                              'enrolled_university', 'education_level', 'major_discipline',
                                              'experience', 'company_size', 'company_type', 'last_new_job',
                                              'training_hours', 'target'])
    ordinal_standard_data = pd.DataFrame(ordinal_standard_data,
                                         columns=['city_development_index', 'gender',
                                                  'relevent_experience', 'enrolled_university', 'education_level',
                                                  'major_discipline', 'experience', 'company_size', 'company_type',
                                                  'last_new_job', 'training_hours', 'target'])
    ordinal_minmax_data = pd.DataFrame(ordinal_minmax_data, columns=['city_development_index', 'gender',
                                                                     'relevent_experience', 'enrolled_university',
                                                                     'education_level', 'major_discipline',
                                                                     'experience', 'company_size', 'company_type',
                                                                     'last_new_job', 'training_hours', 'target'])
    ordinal_robust_data = pd.DataFrame(ordinal_robust_data, columns=['city_development_index', 'gender',
                                                                     'relevent_experience', 'enrolled_university',
                                                                     'education_level', 'major_discipline',
                                                                     'experience', 'company_size', 'company_type',
                                                                     'last_new_job', 'training_hours', 'target'])
    ls_X = label_standard_data.drop(labels="target", axis=1)
    ls_y = label_standard_data["target"]
    lm_X = label_minmax_data.drop(labels="target", axis=1)
    lm_y = label_minmax_data["target"]
    lr_X = label_robust_data.drop(labels="target", axis=1)
    lr_y = label_robust_data["target"]
    #
    os_X = ordinal_standard_data.drop(labels="target", axis=1)
    os_y = ordinal_standard_data["target"]
    om_X = ordinal_minmax_data.drop(labels="target", axis=1)
    om_y = ordinal_minmax_data["target"]
    or_X = ordinal_robust_data.drop(labels="target", axis=1)
    or_y = ordinal_robust_data["target"]


    ##
    # algorithm
    ls_KNN = KNNClassifier(ls_X,ls_y)
    lm_KNN = KNNClassifier(lm_X,lm_y)
    lr_KNN = KNNClassifier(lr_X,lr_y)
    os_KNN = KNNClassifier(os_X, os_y)
    om_KNN = KNNClassifier(om_X, om_y)
    or_KNN = KNNClassifier(or_X, or_y)

    ls_decision = DecisionTree(ls_X, ls_y)
    lm_decision = DecisionTree(lm_X, lm_y)
    lr_decision = DecisionTree(lr_X, lr_y)
    os_decision = DecisionTree(os_X, os_y)
    om_decision = DecisionTree(om_X, om_y)
    or_decision = DecisionTree(or_X, or_y)

    ls_random= random_forest(ls_X, ls_y)
    lm_random= random_forest(lm_X, lm_y)
    lr_random = random_forest(lr_X, lr_y)
    os_random= random_forest(os_X, os_y)
    om_random= random_forest(om_X, om_y)
    or_random = random_forest(or_X, or_y)


    # compare
    # compare = [ls_KNN,lm_KNN,lr_KNN,os_KNN,om_KNN,or_KNN,ls_decision,lm_decision,lr_decision,os_decision,om_decision,or_decision,ls_random,lm_random,lr_random,os_random,om_random,or_random]
    # maxValue = compare[0]
    # for i in range(1,len(compare)):
    #     if maxValue < compare[i]:
    #         maxValue = compare[i]
    compare = {'ls_KNN':ls_KNN, 'lm_KNN':lm_KNN, 'lr_KNN':lr_KNN, 'os_KNN':os_KNN, 'om_KNN':om_KNN, 'or_KNN':or_KNN,
               'ls_decision':ls_decision, 'lm_decision':lm_decision, 'lr_decision':lr_decision,'os_decision':os_decision, 'om_decision':om_decision,'or_decision':or_decision,
               'ls_random':ls_random, 'lm_random':lm_random, 'lr_random':lr_random, 'os_random':os_random, 'om_random':om_random, 'or_random':or_random}

    print("the highest accuracy")
    print(max(compare, key=compare.get))
    print(max(compare.values()))

    return ls_X, ls_y
    # importance_feature(ls_random, x_train, y_train)
    # evaluation(ls_random, x_test, x_train, y_test, y_train, y_pred, model)

def evaluation(x_test, x_train, y_test, y_train, y_pred, model):
    print("Evaluation of ls_random")
    print(classification_report(y_test, y_pred))
    print("ROC AUC: ", roc_auc_score(y_test, y_pred))
    roc_curve_plot(y_test, model.predict_proba(x_test)[:, 1])

    CrossVal(model, x_test, y_test)
    StratKFold(model, x_test, y_test)

    #Grid_Search(x_train, x_test, y_train, y_test)

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



X, y=open()
#float -> int
X = X.astype(int)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

random_forest_model = RandomForestClassifier(max_depth=7, random_state=1)
random_forest_model.fit(X_train, y_train)
y_pred_random_forest = random_forest_model.predict(X_test)
cm_random_forest = confusion_matrix(y_pred_random_forest, y_test)
acc_random_forest = accuracy_score(y_test, y_pred_random_forest)

importance_feature(random_forest_model, X_train, y_train)
evaluation(X_test, X_train, y_test, y_train, y_pred_random_forest, random_forest_model)
# score_KNN = KNNClassifier(X,y)
# print(score_KNN)
#
# random_forest(X, y)
#
# score_dicision = DecisionTree(X,y)
# print(score_dicision)