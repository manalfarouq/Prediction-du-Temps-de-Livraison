
#* fct permet d'imporeter le data
def import_dataset(dataset_name):
    import pandas as pd
    return pd.read_csv(dataset_name)

#* fct permet d'afficher les infos
def affichage_info(dataframe_name):
    return dataframe_name.info()

#* fct permet d'afficher les statistiques
def affichage_dev(dataframe_name):
    return dataframe_name.describe()

#* fct pour vérifier les valeurs manquantes
def isEmpty(dataframe_name):
    return dataframe_name.isnull().sum()

#* fct pour la visualisation avec count_plot
def count_plot_affichage(dataframe_name, column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    sns.countplot(x=column_name, data=dataframe_name)
    plt.title(f"{column_name}")
    plt.xticks(rotation=45)
    plt.show()


#* fct pour la visualisation avec box_plot
def box_plot_affichage(dataframe_name, column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=column_name, y="Delivery_Time_min", data=dataframe_name)
    plt.title(f"{column_name}")
    # plt.xticks(rotation=45)
    plt.show()

#* fct pour séparer le dataset en train/test
def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


#* fct pour encode toutes les colonnes catégorielles spécifiées en valeurs numeriques
def encode_categorical(dataframe_name):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    dataframe_encoded = dataframe_name.copy()
    dataframe_encoded = pd.get_dummies(dataframe_encoded, drop_first=False)
    
    # pour changer seulement les True et false en 0 et 1
    for col in dataframe_encoded.select_dtypes(include=['bool']):
        le = LabelEncoder() 
        dataframe_encoded[col] = le.fit_transform(dataframe_encoded[col])
        
    return dataframe_encoded


#* fct pour selectkbest pour choisir les K (nombre) de colonnes les plus pertinentes pour la prédiction
def select_kbest_columns(dataframe_name):
    from sklearn.feature_selection import SelectKBest, f_regression
    
    X = dataframe_name.drop('Delivery_Time_min', axis=1)
    y = dataframe_name['Delivery_Time_min']
    select = SelectKBest(score_func=f_regression, k=5)
    new_X = select.fit_transform(X,y)
    
    # Récupérer les noms des colonnes sélectionnées
    selected_columns = X.columns[select.get_support()]
    
    return list(selected_columns)

#* fct pour séparer le dataset en train/test
def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= test_size, random_state=random_state)
    return X_train,X_test,y_train,y_test


