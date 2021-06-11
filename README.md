**Open Source function** 

**user manual / specification** 

--------------------------------------------------------------------------------------------

: combination of numerical data scaling and categorical data encoding for a machine learning algorithm



* Encoding includes label encoder and ordinary encoder.
  Scaling includes standard scaling, minmax scaling, and robustscaling.
  Algorithms include decision tree, random forest, and knn.

* Used Methods : cal_score, encoding, scaling, algorithmmethod, DecisionTree, random_forest, KNNClassifier
* Method Details




| __Method__                            | __Description__                                              |
| ------------------------------------- | ------------------------------------------------------------ |
| OpenSource(encode, scaler, algorithm) | __Parameters:__ encode, scaler, algorithm__Returns:__ encode_list[index_value], scaler_list[index_value], algorithm_list[index_value]__Description:__ combination of numerical data __scaling__ and categorical data __encoding__ for a machine learning __algorithm__.All parameter values are a list and are as follows.`</p><p>encode = ['label', 'ordinal']</p><p>scaler = ['standard', 'minmax', 'robust']</p><p>algorithm = ['decision', 'random', 'knn']</p><p>`For cal_score, run a triple for statement to get each combination value.When the score value is obtained through the cal_score function, it is added to the compare list. The encoding, scaler, and algorithm corresponding to the score are also included in each list. After that, the optimal combination and max are found. |
| cal_score(enocode, scaler, algorithm) | __Parameters:__ encode, scaler, algorithm__Returns:__ score__Description:__ Combination of encoding, scaler, and algorithm calculates the accuracy score value. The functions used at this time include encoding, scaling, and algorithmmethod. |
| encoding(encode)                      | __Parameters:__ encode__Returns:__ encode_data__Description:__ It compares the encoding received as a parameter with the string of the two encodings, and returns the encoded data by using labelEncoder or OrdinalEncoder. |
| scaling(encode, scaler)               | __Parameters:__ encode, scaler__Returns:__ encode__Description:__ It compares the string of three scalers with the scaler received as a parameter, and returns the scaled data by using StandardScaler, MinMaxScaler, or RobustScaler. |
| algorithmmethod(algorithm, X, y)      | __Parameters:__ algorithm, X, y__Returns:__ score__Description:__ By comparing the algorithm received as a parameter and the string of three algorithms, the corresponding methods are called to do DecisionTree or random_forest or KNNClassifier. And the corresponding accuracy score is obtained. |
| DecisionTree(X, y)                    | __Parameters:__ X, y__Returns:__ accuracy_score(y_test, dtc.predict(X_test))__Description:__ It is a method that does DecisionTreeClassifier, and gets accuracy_score that does DecisionTree. |
| random_forest(X, y)                   | __Parameters:__ X, y__Returns:__ accuracy_score(y_test, random_forest_model.predict(X_test))__Description:__ It is a method that does RandomForestClassifier, and gets accuracy_score that does random forest. |
| KNNClassifier(X, y)                   | __Parameters:__ X, y__Returns:__ accuracy_score(y_test, knn.predict(X_test))__Description:__ It is a method that does KNeighborsClassifier, and gets accuracy_score with knn. |
