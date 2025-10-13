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
    sns.countplot(x=column_name, data=dataframe_name, hue='Delivery_Time_min')
    plt.title(f"{column_name}")
    plt.xticks(rotation=45)
    plt.show()



# fct pour la visualisation avec hist_plot
def hist_plot_affichage(dataframe_name, column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    sns.histplot(x=column_name, data=dataframe_name, hue='Delivery_Time_min')
    plt.title(f"{column_name}")
    plt.xticks(rotation=45)
    plt.show()
