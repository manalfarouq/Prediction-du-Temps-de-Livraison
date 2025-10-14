# fct permet d'imporeter le data
def import_dataset(dataset_name):
    import pandas as pd
    return pd.read_csv(dataset_name)

# fct permet d'afficher les infos
def affichage_info(dataframe_name):
    return dataframe_name.info()

# fct permet d'afficher les statistiques
def affichage_dev(dataframe_name):
    return dataframe_name.describe()

# fct pour vérifier les valeurs manquantes
def isEmpty(dataframe_name):
    return dataframe_name.isnull().sum()

# fct pour la visualisation avec count_plot
def count_plot_affichage(dataframe_name, column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    sns.countplot(x=column_name, data=dataframe_name)
    plt.title(f"{column_name}")
    plt.xticks(rotation=45)
    plt.show()


# fct pour la visualisation avec box_plot
def box_plot_affichage(dataframe_name, column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=column_name, y="Delivery_Time_min", data=dataframe_name)
    plt.title(f"{column_name}")
    # plt.xticks(rotation=45)
    plt.show()

# fct pour séparer le dataset en train/test
def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# fct pour encode toutes les colonnes catégorielles spécifiées en valeurs numeriques
def encode_categorical(dataframe_name):
    from sklearn.preprocessing import LabelEncoder

    dataframe_encoded = dataframe_name.copy()
    categorical_cols = dataframe_encoded.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        dataframe_encoded[col] = le.fit_transform(dataframe_encoded[col])
    return dataframe_encoded
