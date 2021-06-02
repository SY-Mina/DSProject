import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import itertools
from sklearn.preprocessing import FunctionTransformer

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

for c in categorical_list:
    bar_plot(c)

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
    plt.show()

for n in numerical_int64_list:
    plot_hist(n)

numerical_float64 = (dataset.dtypes == "float64")
numerical_float64_list = list(numerical_float64[numerical_float64].index)

print("Numerical variables:")
print(numerical_float64_list)

for n in numerical_float64_list:
    plot_hist(n)


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

#나머지 데이터들 인코딩
cat_cols = ['gender', 'enrolled_university', 'major_discipline', 'company_type']
labelEncoder = LabelEncoder()
for col in cat_cols:
    dataset[col] = labelEncoder.fit_transform(dataset[col])

print(dataset.head(15))



