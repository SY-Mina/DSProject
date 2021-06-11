\# Open Source function 

\## user manual / specification 

\---------------------------

: combination of numerical data \_\_scaling\_\_ and categorical data \_\_encoding\_\_ for a machine learning \_\_algorithm\_\_

\* Encoding includes label encoder and ordinary encoder.

`  `Scaling includes standard scaling, minmax scaling, and robustscaling.

`  `Algorithms include decision tree, random forest, and knn.

Used Methods : cal\_score, encoding, scaling, algorithmmethod, DecisionTree, random\_forest, KNNClassifier

| \_\_Method\_\_                         | \_\_Description\_\_                                          |
| :------------------------------------- | :----------------------------------------------------------- |
| OpenSource(encode, scaler, algorithm)  | <p>\_\_Parameters:\_\_ encode, scaler, algorithm</p><p>\_\_Returns:\_\_ encode\_list[index\_value], scaler\_list[index\_value], algorithm\_list[index\_value]</p><p>\_\_Description:\_\_ combination of numerical data \_\_scaling\_\_ and categorical data \_\_encoding\_\_ for a machine learning \_\_algorithm\_\_.</p><p>All parameter values are a list and are as follows.</p><p></p><p>```</p><p>encode = ['label', 'ordinal']</p><p>scaler = ['standard', 'minmax', 'robust']</p><p>algorithm = ['decision', 'random', 'knn']</p><p>```</p><p>For cal\_score, run a triple for statement to get each combination value.</p><p>When the score value is obtained through the cal\_score function, it is added to the compare list. The encoding, scaler, and algorithm corresponding to the score are also included in each list. After that, the optimal combination and max are found.</p> |
| cal\_score(enocode, scaler, algorithm) | <p>\_\_Parameters:\_\_ encode, scaler, algorithm</p><p>\_\_Returns:\_\_ score</p><p>\_\_Description:\_\_ Combination of encoding, scaler, and algorithm calculates the accuracy score value. The functions used at this time include encoding, scaling, and algorithmmethod.</p><p></p> |
| encoding(encode)                       | <p>\_\_Parameters:\_\_ encode</p><p>\_\_Returns:\_\_ encode\_data</p><p>\_\_Description:\_\_ It compares the encoding received as a parameter with the string of the two encodings, and returns the encoded data by using labelEncoder or OrdinalEncoder.</p><p></p> |
| scaling(encode, scaler)                | <p>\_\_Parameters:\_\_ encode, scaler</p><p>\_\_Returns:\_\_ encode</p><p>\_\_Description:\_\_ It compares the string of three scalers with the scaler received as a parameter, and returns the scaled data by using StandardScaler, MinMaxScaler, or RobustScaler.</p><p></p> |
| algorithmmethod(algorithm, X, y)       | <p>\_\_Parameters:\_\_ algorithm, X, y</p><p>\_\_Returns:\_\_ score</p><p>\_\_Description:\_\_ By comparing the algorithm received as a parameter and the string of three algorithms, the corresponding methods are called to do DecisionTree or random\_forest or KNNClassifier. And the corresponding accuracy score is obtained.</p><p></p> |
| DecisionTree(X, y)                     | <p>\_\_Parameters:\_\_ X, y</p><p>\_\_Returns:\_\_ accuracy\_score(y\_test, dtc.predict(X\_test))</p><p>\_\_Description:\_\_ It is a method that does DecisionTreeClassifier, and gets accuracy\_score that does DecisionTree.</p><p></p> |
| random\_forest(X, y)                   | <p>\_\_Parameters:\_\_ X, y</p><p>\_\_Returns:\_\_ accuracy\_score(y\_test, random\_forest\_model.predict(X\_test))</p><p>\_\_Description:\_\_ It is a method that does RandomForestClassifier, and gets accuracy\_score that does random forest.</p><p></p> |
| KNNClassifier(X, y)                    | <p>\_\_Parameters:\_\_ X, y</p><p>\_\_Returns:\_\_ accuracy\_score(y\_test, knn.predict(X\_test))</p><p>\_\_Description:\_\_ It is a method that does KNeighborsClassifier, and gets accuracy\_score with knn.</p><p></p> |

