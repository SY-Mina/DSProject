**Open Source function** 

**user manual / specification** 

--------------------------------------------------------------------------------------------

: combination of numerical data scaling and categorical data encoding for a machine learning algorithm



Encoding includes label encoder and ordinary encoder.
Scaling includes standard scaling, minmax scaling, and robustscaling.
Algorithms include decision tree, random forest, and knn.

Used Methods : cal_score, encoding, scaling, algorithmmethod, DecisionTree, random_forest, KNNClassifier

| **Method**                             | **Description**                                              |
| :------------------------------------- | :----------------------------------------------------------- |
| OpenSource(encode, scaler, algorithm)  | <p>**Parameters:** encode, scaler, algorithm</p><p>**Returns:** encode\_list[index\_value], scaler\_list[index\_value], algorithm\_list[index\_value]</p><p>**Description:** combination of numerical data scaling and categorical data encoding for a machine learning algorithm.</p><p>All parameter values are a list and are as follows.</p><p></p><p>encode = ['label', 'ordinal']</p><p>scaler = ['standard', 'minmax', 'robust']</p><p>algorithm = ['decision', 'random', 'knn']</p><p></p><p>For cal\_score, run a triple for statement to get each combination value.</p><p>When the score value is obtained through the cal\_score function, it is added to the compare list. The encoding, scaler, and algorithm corresponding to the score are also included in each list. After that, the optimal combination and max are found.</p> |
| cal\_score(enocode, scaler, algorithm) | <p>**Parameters:** encode, scaler, algorithm</p><p>**Returns:** score</p><p>**Description:** Combination of encoding, scaler, and algorithm calculates the accuracy score value. The functions used at this time include encoding, scaling, and algorithmmethod.</p><p></p> |
| encoding(encode)                       | <p>**Parameters:** encode</p><p>**Returns:** encode\_data</p><p>**Description:** It compares the encoding received as a parameter with the string of the two encodings, and returns the encoded data by using labelEncoder or OrdinalEncoder.</p><p></p> |
| scaling(encode, scaler)                | <p>**Parameters:** encode, scaler</p><p>**Returns:** encode</p><p>**Description:** It compares the string of three scalers with the scaler received as a parameter, and returns the scaled data by using StandardScaler, MinMaxScaler, or RobustScaler.</p><p></p> |
| algorithmmethod(algorithm, X, y)       | <p>**Parameters:** algorithm, X, y</p><p>**Returns:** score</p><p>**Description:** By comparing the algorithm received as a parameter and the string of three algorithms, the corresponding methods are called to do DecisionTree or random\_forest or KNNClassifier. And the corresponding accuracy score is obtained.</p><p></p> |
| DecisionTree(X, y)                     | <p>**Parameters:** X, y</p><p>**Returns:** accuracy\_score(y\_test, dtc.predict(X\_test))</p><p>**Description:** It is a method that does DecisionTreeClassifier, and gets accuracy\_score that does DecisionTree.</p><p></p> |
| random\_forest(X, y)                   | <p>**Parameters:** X, y</p><p>**Returns:** accuracy\_score(y\_test, random\_forest\_model.predict(X\_test))</p><p>**Description:** It is a method that does RandomForestClassifier, and gets accuracy\_score that does random forest.</p><p></p> |
| KNNClassifier(X, y)                    | <p>**Parameters:** X, y</p><p>**Returns:** accuracy\_score(y\_test, knn.predict(X\_test))</p><p>**Description:** It is a method that does KNeighborsClassifier, and gets accuracy\_score with knn.</p><p></p> |

