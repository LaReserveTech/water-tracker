from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests
from tqdm import tqdm
from scipy.stats import lognorm
import spei as si
import scipy.stats as scs

class WaterTracker():
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_folder = "data/"
        #self.timeseries_folder = self.data_folder+"timeseries/"
        #self.df_stations = pd.read_csv(f"{self.data_folder}df_stations.csv")
        self.mapping_indicateur_column = {"pluviométrie":"dryness-meteo", 
                                          "nappes": "dryness-groundwater",
                                          "débit": "dryness-stream-flow"
                                          }

        self.mapping_indicator_names = {"dryness-meteo":"rain-level",
                                        "dryness-groundwater":"water-level-static",
                                        "dryness-stream-flow":"stream-flow"
                                        }


        # Some stations have strange values, we get rid off them
        self.black_listed_station_ids = [1951]


    def download_departement_data(self, departement_code):
            headers = {'accept': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
            params = {'with': 'geometry;indicators.state.type;locations.type;locations.indicators.state.type'}

            try:
                response = requests.get(f'https://api.emi.imageau.eu/app/departments/{departement_code}', params=params, headers=headers)
            except Exception as e:
                print(e)
                return None
            else:
                if response.status_code == 200:
                    return dict(response.json())
                else:
                    print(f"Request failed with status code {response.status_code}")
                    return None 

    def download_stations_data(self):
        stations_data = {}
        for i in tqdm(['2A', '2B'] + list(range(1, 20)) + list(range(21, 96))):
            stations_data[i] = self.download_departement_data(str(i).rjust(2, "0"))
        return stations_data

    def build_stations_data(self):
        """
        Call this function to create the df_stations.csv file storing all data about stations that are needed for further computations (timeseries, computation of standardized indicators, plots)
        """
        stations_data = self.download_stations_data()
        res = []
        for departement_code in stations_data.keys():
            for station in stations_data[departement_code]["data"]["locations"]:
                data = {"departement_code" : departement_code,
                        "id": station["id"],
                        'name': station["name"],
                        'bss_code': station["bss_code"],  # Utilisez la clé bss_code sans répétition
                        "indicators": [indicator["state"]["type"]["name"] for indicator in station["indicators"]],
                    }
                res.append(data)

        # Créez le dossier 'data' s'il n'existe pas
        data_folder = os.path.abspath("./data")
        os.makedirs(data_folder, exist_ok=True)

        timeseries_folder = os.path.join(data_folder, "timeseries")
        os.makedirs(timeseries_folder, exist_ok=True)

        df_stations = pd.DataFrame(res)
        df_stations = pd.concat([df_stations, pd.get_dummies(df_stations["indicators"].explode()).groupby(level=0).sum()], axis=1).drop("indicators", axis=1)
        output_filename = os.path.abspath("./data/df_stations.csv")
        print(f"Sauvegarde des données des stations dans {output_filename}")
        df_stations.to_csv(output_filename, index=False)
        self.timeseries_folder = self.data_folder+"timeseries/"
        self.df_stations = pd.read_csv(f"{self.data_folder}df_stations.csv")

    

    def download_timeseries_station(self, location_id, start_date, end_date):
        headers = {'accept': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        params = {'location_id': str(location_id), 'from': start_date, 'to': end_date}
        
        try:
            response = requests.get('https://api.emi.imageau.eu/app/data', params=params, headers=headers)
        except Exception as e:
            print(e)
            return
        else:
            if response.status_code == 200:
                return dict(response.json())
            else:
                return None


    def download_all_timeseries(self):
        today = datetime.today().strftime('%Y-%m-%d')
        current_month = datetime.today().strftime('%Y-%m')
        df = pd.read_csv(os.path.join(self.data_folder, "df_stations.csv"))
        cpt = 0
        n = len(df)
        os.makedirs(self.timeseries_folder, exist_ok=True)

        for i, row in df.iterrows():
            station_id = row["id"]
            if station_id not in self.black_listed_station_ids:
                cpt += 1
                print(f"{100 * cpt / n}%", end="\r")
                indicator_name = None

                if row["dryness-stream-flow"] == 1:
                    indicator_name = self.mapping_indicator_names["dryness-stream-flow"]
                elif row["dryness-meteo"] == 1:
                    indicator_name = self.mapping_indicator_names["dryness-meteo"]
                elif row["dryness-groundwater"] == 1:
                    indicator_name = self.mapping_indicator_names["dryness-groundwater"]
                else:
                    continue

                indicator_folder = os.path.join(self.timeseries_folder, indicator_name)
                os.makedirs(indicator_folder, exist_ok=True)  # Créer le sous-dossier si inexistant

                filename = os.path.join(indicator_folder, f"{station_id}.csv")
                d = None

                if not os.path.isfile(filename):
                    # Téléchargez toutes les données disponibles pour chaque station
                    d = self.download_timeseries_station(station_id, "1970-01-01", today)

                    # Vérifiez si la réponse n'est pas None
                    if d is not None and indicator_name in d and len(d[indicator_name]) >= 15 * 365:
                        # Vérifier si les données pour le mois en cours sont présentes
                        current_month_data = [entry for entry in d[indicator_name] if entry['date'][:7] == current_month]
                        if not current_month_data:
                            print(f"La station {station_id} n'a pas de données pour le mois en cours. Les données ne seront pas téléchargées.")
                            continue

                        timeseries = pd.DataFrame(d[indicator_name])[["date", "value"]]
                        timeseries["date"] = pd.to_datetime(timeseries["date"])
                        timeseries = timeseries.drop_duplicates()
                        timeseries.to_csv(filename)
                    else:
                        print(f"La station {station_id} n'a pas suffisamment de données sur 15 ans. Les données ne seront pas téléchargées.")
                        continue

                timeseries_init = pd.read_csv(filename)
                timeseries_init["date"] = pd.to_datetime(timeseries_init["date"])
                next_date = (timeseries_init["date"].max().to_pydatetime() + timedelta(days=1)).date()
                d = self.download_timeseries_station(station_id, next_date, today)

                if indicator_name in d and len(d[indicator_name]) > 0:
                    timeseries = pd.DataFrame(d[indicator_name])[["date", "value"]]
                    timeseries["date"] = pd.to_datetime(timeseries["date"])
                    timeseries = timeseries.drop_duplicates()
                    timeseries_final = pd.concat([timeseries_init, timeseries], axis=0)
                    timeseries_final.to_csv(filename)




#Calcul de l'indice sécheresse avec les débits de cours d'eau


    def apply_galton_law_to_single_station(self, station_id):
        # Vérifier si le fichier station_id.csv existe dans le dossier timeseries
        filename = f"./data/timeseries/stream-flow/{station_id}.csv"
        if os.path.isfile(filename):
            timeseries = pd.read_csv(filename)

            # Assurez-vous que la colonne "date" est de type datetime
            timeseries["date"] = pd.to_datetime(timeseries["date"])

            # Assurez-vous que la colonne "value" est numérique
            timeseries["value"] = pd.to_numeric(timeseries["value"], errors='coerce')

            # Vérifiez si la colonne "value" est présente et a suffisamment de données
            if "value" in timeseries.columns and len(timeseries) >= 15 * 365:
                timeseries.set_index("date", inplace=True)

                # Calcul du VCN3 (moyenne sur 3 jours)
                vcn3 = timeseries["value"].rolling(window=3).mean()
                # Calcul du VCN3 (minimum mensuel de la moyenne sur 3 jours)
                minvcn3 = timeseries["value"].rolling(window=3).mean().resample("M").min()

                # Création d'un DataFrame pour stocker les résultats
                minvcn3_df = pd.DataFrame({"Month": minvcn3.index.month, "Year": minvcn3.index.year, "VCN3": minvcn3.values})
                # Sauvegarder minvcn3_df dans un fichier CSV
                #minvcn3_df.to_csv(f"./minvcn3_df_{station_id}.csv", index=False)

                # Création d'un DataFrame pour stocker les résultats
                vcn3_df = pd.DataFrame({"Date": vcn3.index + pd.DateOffset(1), "VCN3": vcn3.values})
                # Sauvegarder vcn3_df dans un fichier CSV
                #vcn3_df.to_csv(f"./vcn3_df_{station_id}.csv", index=False)
                # Convertir la colonne "Date" au format datetime
                vcn3_df["Date"] = pd.to_datetime(vcn3_df["Date"], format="%Y-%m-%d")
                vcn3_df.set_index("Date", inplace=True)
                vcn3_values = vcn3_df["VCN3"].dropna()

                # Sauvegarder minvcn3_df dans un fichier CSV
                #minvcn3_df.to_csv("./minvcn3_df.csv", index=False)

                # Ajouter une colonne "date" basée sur les colonnes "Year" et "Month"
                minvcn3_df["date"] = pd.to_datetime(minvcn3_df[["Year", "Month"]].assign(DAY=1))
                minvcn3_df.set_index("date", inplace=True)
                minvcn3_values = minvcn3_df["VCN3"].dropna()

                # Initialiser le DataFrame résultant
                result_df = pd.DataFrame(columns=["Date", "VCN3", "CDF", "Level"])

                # Appliquer la loi de Galton mois par mois
                for month in range(1, 13):
                    # Filtrer les valeurs pour le mois actuel
                    minmonth_values = minvcn3_values[minvcn3_values.index.month == month]
                    month_values = vcn3_values[vcn3_values.index.month == month]

                    if len(minmonth_values) > 0:
                        # Ajuster une distribution lognormale pour le mois actuel
                        params = lognorm.fit(minmonth_values)
                        dist = lognorm(*params)

                        # Calculer la fonction de répartition cumulative (CDF) pour chaque valeur VCN3
                        cdf_values = dist.cdf(month_values)

                        # Associer un niveau à chaque probabilité cumulative
                        levels = pd.cut(cdf_values, bins=[0, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 1.0],
                                        labels=["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"])
                        # Convertir les niveaux en chaînes correctement encodées
                        #levels = levels.astype(str).apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))


                        # Créer un DataFrame pour le mois actuel
                        month_df = pd.DataFrame({
                            "Date": month_values.index,
                            "VCN3": month_values.values,
                            "CDF": cdf_values,
                            "Level": levels
                        })

                        # Ajouter le DataFrame du mois à la liste
                        result_df = pd.concat([result_df, month_df], ignore_index=True)

                # Trier le DataFrame par date
                result_df.sort_values(by="Date", inplace=True)

                # Réinitialiser les index
                result_df.reset_index(drop=True, inplace=True)
                # Compléter les dates manquantes
                result_df = self.complete_missing_months(result_df)
                # Sélectionner la date du jour moins un an en arrière
                #one_year_ago = result_df['Date'].max() - pd.DateOffset(months=12)


                # Filtrer les données pour les douze derniers mois
                #result_df = result_df[result_df['Date'] >= one_year_ago]


                # Sauvegarder le DataFrame résultant (en remplaçant le fichier s'il existe)
                #output_folder = "./data2/galton_results/"
                #os.makedirs(output_folder, exist_ok=True)
                #result_filename = os.path.join(output_folder, f"{station_id}_galton_results.csv")

                #if os.path.exists(result_filename):
                    #result_df.to_csv(result_filename, index=False, mode='w', header=True)
                #else:
                    #result_df.to_csv(result_filename, index=False, mode='w', header=True)

                return result_df




    


    def complete_missing_months(self, df):

        all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq='D')

        # Réinitialiser l'index du DataFrame complet
        all_dates_df = pd.DataFrame({"Date": all_dates})
        all_dates_df.reset_index(drop=True, inplace=True)

        result_df = pd.merge(all_dates_df, df, on="Date", how="left")

        return result_df



    def apply_galton_law_to_all_stations(self):
        data_folder = "./data/timeseries/stream-flow/"
        timeseries_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

        if not timeseries_files:
            print("Aucun fichier de séries temporelles trouvé.")
            return pd.DataFrame()

        result_df_all_stations = pd.DataFrame(columns=["Date", "VCN3", "CDF", "Level"])

        for station_file in timeseries_files:
            # Extraire l'ID de la station à partir du nom du fichier
            station_id = station_file.split(".")[0]

            # Appliquer la loi de Galton à une station spécifique
            station_result_df = self.apply_galton_law_to_single_station(station_id)

            if not station_result_df.empty:
                # Ajouter les résultats de la station au DataFrame résultant
                result_df_all_stations = pd.concat([result_df_all_stations, station_result_df], ignore_index=True)
            else:
                print(f"Le traitement de la station {station_id} n'a pas abouti.")

        result_df_all_stations.sort_values(by="Date", inplace=True)

        result_df_all_stations.reset_index(drop=True, inplace=True)
        # Compléter les dates manquantes
        result_df_all_stations = self.complete_missing_months(result_df_all_stations)
        # Sélectionner la date du jour moins un an en arrière
        one_year_ago = result_df_all_stations['Date'].max() - pd.DateOffset(months=12)

        # Filtrer les données pour les douze derniers mois. Decommentez la ligne suivante
        #result_df_all_stations = result_df_all_stations[result_df_all_stations['Date'] >= one_year_ago]
        # Sélectionner les données de l'année 2023
        result_df_all_stations = result_df_all_stations[(result_df_all_stations['Date'] >= '2023-01-01') & (result_df_all_stations['Date'] <= '2023-12-31')]


        # Sauvegarder le DataFrame résultant dans un fichier pickle
        #pickle_filename = "./data2/galton_results/result_df_all_stations.pkl"
        #result_df_all_stations.to_pickle(pickle_filename)
        #print(f"DataFrame sauvegardé dans {pickle_filename}")

        return result_df_all_stations
    





    def plot_level_debit(self, result_df_all_stations):

        result_df_all_stations['Date'] = pd.to_datetime(result_df_all_stations['Date'])

        # Ajouter une colonne 'YearMonth' basée sur les colonnes 'Year' et 'Month'
        result_df_all_stations['YearMonth'] = result_df_all_stations['Date'].dt.to_period('M')

        # Mapping personnalisé des noms de mois en français
        month_mapping = {
                    "01": "Janvier",
                    "02": "Février",
                    "03": "Mars",
                    "04": "Avril",
                    "05": "Mai",
                    "06": "Juin",
                    "07": "Juillet",
                    "08": "Août",
                    "09": "Septembre",
                    "10": "Octobre",
                    "11": "Novembre",
                    "12": "Décembre"
                }

        # Ignorer les lignes où 'Level' est manquant
        result_df_all_stations = result_df_all_stations.dropna(subset=['Level'])

        # Calculer le pourcentage des niveaux par mois
        percentage_by_month = (
            result_df_all_stations.groupby(['YearMonth', 'Level']).size() /
            result_df_all_stations.groupby('YearMonth').size() * 100
        )

        # Reformater les données pour le tracé
        percentage_by_month = percentage_by_month.unstack('Level')

        # Vérifier s'il y a treize mois
        if len(percentage_by_month) == 13:
            # Ignorer la première ligne (premier mois)
            percentage_by_month = percentage_by_month.iloc[1:]

        # Tri des colonnes selon l'ordre spécifié
        level_order = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        percentage_by_month = percentage_by_month[level_order]
        percentage_by_month = percentage_by_month.reset_index().rename(columns={'YearMonth': 'Mois'})
        percentage_by_month['Mois'] = percentage_by_month['Mois'].apply(lambda x: month_mapping[x.strftime('%m')] + ' ' + x.strftime('%Y'))



        # Couleurs
        colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

        # Tracé
        ax = percentage_by_month.plot(kind='bar', stacked=True, color=colors)

        # Ajustements esthétiques
        plt.yticks(range(0, 101, 10))
        plt.title('Débit: Répartition des niveaux de sécheresse en 2023')
        plt.xlabel('')
        plt.ylabel('Proportion des stations (%)')

        # Utiliser les noms de mois traduits sur l'axe des abscisses
        ax.set_xticklabels(percentage_by_month['Mois'], rotation=45, ha='right')

        # Légende
        legend_labels = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

        # Réglage automatique de l'orientation des dates sur l'axe des x
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=45)

        # Sauvegarde du graphique
        save_directory="./images/"
        os.makedirs(save_directory, exist_ok=True)
        image_filename = './images/debit.pdf'
        print(f"Sauvegarde du graphique dans {image_filename}")
        fig.savefig(image_filename, bbox_inches='tight')
        # Sauvegarde des données
        data_filename = f'./data/débit_data.csv'
        print(f"Saving data to {data_filename}")
        percentage_by_month .to_csv(data_filename, index=False)
        #plt.show()


    def spi_to_single_station(self, station_id):
        data_folder = "./data/timeseries/rain-level/"
        #save_directory = "./data/timeseries/spi-results/"
        #os.makedirs(save_directory, exist_ok=True)
        filename = f"{data_folder}/{station_id}.csv"
        if os.path.isfile(filename):
            timeseries = pd.read_csv(filename)
                
            if "value" in timeseries.columns and len(timeseries) >= 15 * 365:
                timeseries.set_index("date", inplace=True)


            timeseries_series = timeseries["value"].rolling(window=30).sum().dropna()


            # Calcul du SPI avec la distribution gamma
            spi_result = si.spi(timeseries_series, dist=scs.gamma).rename("spi")
        
            spi_result = pd.DataFrame(spi_result)
            spi_result["date"] = spi_result.index

            spi_result.rename(columns={"date": "Date"}, inplace=True)
            #print(spi_result)
            # Définir les niveaux de SPI
            bins = [-float('inf'), -2.00, -1.49, -0.99, 0.99, 1.49, 2.00, float('inf')]
            spi_levels = ["Sécheresse extrême", "Grande sécheresse", "Sécheresse modérée","Situation normale", "Modérément humide", "Très humide", "Extrêmement humide"]

            spi_result['Level'] = pd.cut(spi_result['spi'], bins=bins, labels=spi_levels, right=False)

            spi_result = self.complete_missing_months(spi_result)


            # Sélectionner la date du jour moins un an en arrière
            one_year_ago = spi_result['Date'].max() - pd.DateOffset(months=12)

            # Filtrer les données pour les douze derniers mois. Decommentez la ligne suivante
            #spi_result = spi_result[spi_result['Date'] >= one_year_ago]
            # Sélectionner les données de l'année 2023
            spi_result = spi_result[(spi_result['Date'] >= '2023-01-01') & (spi_result['Date'] <= '2023-12-31')]

            #filename = f"{save_directory}/{station_id}_sgi_result.csv"
            #sgi_result.to_csv(filename)
            #print(f"Le SGI résultant pour la station {station_id} a été sauvegardé dans {filename}")

            return spi_result

    def spi_to_all_stations(self):
        data_folder = "./data/timeseries/rain-level/"
        #save_directory = "./data/timeseries/spi-results/"
        #os.makedirs(save_directory, exist_ok=True)
        station_ids = [file.split(".")[0] for file in os.listdir(data_folder) if file.endswith(".csv")]
        spi_result_all_stations = pd.DataFrame()

        for station_id in station_ids:
            station_spi_result = self.spi_to_single_station(station_id)
            if not station_spi_result.empty:
                spi_result_all_stations = pd.concat([spi_result_all_stations, station_spi_result], ignore_index=True)

        #spi_result_all_stations.to_pickle("./data/timeseries/spi-results/spi_result_all_stations.pkl")
        #print("Le DataFrame résultant pour toutes les stations a été sauvegardé dans spi_result_all_stations.pkl")

        return spi_result_all_stations


    def sgi_to_single_station(self, station_id):
        data_folder = "./data/timeseries/water-level-static/"
        #save_directory = "./data/timeseries/sgi-results/"
        #os.makedirs(save_directory, exist_ok=True)
        filename = f"{data_folder}/{station_id}.csv"
        if not os.path.isfile(filename):
            print(f"Le fichier {station_id}.csv n'existe pas.")
            return pd.DataFrame()

        try:
            timeseries = pd.read_csv(filename)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {station_id}.csv : {e}")
            return pd.DataFrame()

        if "value" not in timeseries.columns or len(timeseries) < 15 * 365:
            print(f"Le fichier {station_id}.csv ne contient pas de colonne 'value' ou n'a pas assez de données.")
            return pd.DataFrame()

        try:
            timeseries["date"] = pd.to_datetime(timeseries["date"])
        except Exception as e:
            print(f"Erreur lors de la conversion de la colonne 'date' en datetime : {e}")
            return pd.DataFrame()

        timeseries = timeseries[~timeseries.duplicated(subset="date", keep='first')]

        timeseries_series = pd.Series(timeseries["value"].values, index=timeseries["date"])

        try:
            sgi_result = si.sgi(timeseries_series).rename("sgi")
        except Exception as e:
            print(f"Erreur lors du calcul du SGI : {e}")
            return pd.DataFrame()

        sgi_result = pd.DataFrame(sgi_result)
        sgi_result["date"] = sgi_result.index
        sgi_result.rename(columns={"date": "Date"}, inplace=True)

        bins = [-float('inf'), -1.28, -0.84, -0.25, 0.25, 0.84, 1.28, float('inf')]
        sgi_levels = ["Très bas", "Bas", "Modérément bas","Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        sgi_result['Level'] = pd.cut(sgi_result['sgi'], bins=bins, labels=sgi_levels, right=False)

        sgi_result = self.complete_missing_months(sgi_result)
        # Sélectionner la date du jour moins un an en arrière
        one_year_ago = sgi_result['Date'].max() - pd.DateOffset(months=12)

        # Filtrer les données pour les douze derniers mois. Decommentez la ligne suivante
        #sgi_result = sgi_result[sgi_result['Date'] >= one_year_ago]
        # Sélectionner les données de l'année 2023
        sgi_result = sgi_result[(sgi_result['Date'] >= '2023-01-01') & (sgi_result['Date'] <= '2023-12-31')]

        #filename = f"{save_directory}/{station_id}_sgi_result.csv"
        #sgi_result.to_csv(filename)
        #print(f"Le SGI résultant pour la station {station_id} a été sauvegardé dans {filename}")

        return sgi_result

    def sgi_to_all_stations(self):
        data_folder = "./data/timeseries/water-level-static/"
        #save_directory = "./data/timeseries/sgi-results/"
        #os.makedirs(save_directory, exist_ok=True)
        station_ids = [file.split(".")[0] for file in os.listdir(data_folder) if file.endswith(".csv")]
        sgi_result_all_stations = pd.DataFrame()

        for station_id in station_ids:
            station_sgi_result = self.sgi_to_single_station(station_id)
            if not station_sgi_result.empty:
                sgi_result_all_stations = pd.concat([sgi_result_all_stations, station_sgi_result], ignore_index=True)

        #sgi_result_all_stations.to_pickle("./data/timeseries/sgi-results/sgi_result_all_stations.pkl")
        #print("Le DataFrame résultant pour toutes les stations a été sauvegardé dans sgi_result_all_stations.pkl")

        return sgi_result_all_stations


    def plot_level_nappes(self, sgi_result_all_stations):

        sgi_result_all_stations['Date'] = pd.to_datetime(sgi_result_all_stations['Date'])

        # Ajouter une colonne 'YearMonth' basée sur les colonnes 'Year' et 'Month'
        sgi_result_all_stations['YearMonth'] = sgi_result_all_stations['Date'].dt.to_period('M')

        # Mapping personnalisé des noms de mois en français
        month_mapping = {
                    "01": "Janvier",
                    "02": "Février",
                    "03": "Mars",
                    "04": "Avril",
                    "05": "Mai",
                    "06": "Juin",
                    "07": "Juillet",
                    "08": "Août",
                    "09": "Septembre",
                    "10": "Octobre",
                    "11": "Novembre",
                    "12": "Décembre"
                }

        # Ignorer les lignes où 'Level' est manquant
        sgi_result_all_stations = sgi_result_all_stations.dropna(subset=['Level'])

        # Calculer le pourcentage des niveaux par mois
        percentage_by_month = (
            sgi_result_all_stations.groupby(['YearMonth', 'Level']).size() /
            sgi_result_all_stations.groupby('YearMonth').size() * 100
        )

        # Reformater les données pour le tracé
        percentage_by_month = percentage_by_month.unstack('Level')

        # Vérifier s'il y a treize mois
        if len(percentage_by_month) == 13:
            # Ignorer la première ligne (premier mois)
            percentage_by_month = percentage_by_month.iloc[1:]

        # Tri des colonnes selon l'ordre spécifié
        level_order = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        percentage_by_month = percentage_by_month[level_order]
        percentage_by_month = percentage_by_month.reset_index().rename(columns={'YearMonth': 'Mois'})
        percentage_by_month['Mois'] = percentage_by_month['Mois'].apply(lambda x: month_mapping[x.strftime('%m')] + ' ' + x.strftime('%Y'))



        # Couleurs
        colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

        # Tracé
        ax = percentage_by_month.plot(kind='bar', stacked=True, color=colors)

        # Ajustements esthétiques
        plt.yticks(range(0, 101, 10))
        plt.title('Nappes: Répartition des niveaux de sécheresse en 2023')
        plt.xlabel('')
        plt.ylabel('Proportion des stations (%)')

        # Utiliser les noms de mois traduits sur l'axe des abscisses
        ax.set_xticklabels(percentage_by_month['Mois'], rotation=45, ha='right')

        # Légende
        legend_labels = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

        # Réglage automatique de l'orientation des dates sur l'axe des x
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=45)

        # Sauvegarde du graphique
        save_directory="./images/"
        os.makedirs(save_directory, exist_ok=True)
        image_filename = './images/nappes.pdf'
        print(f"Sauvegarde du graphique dans {image_filename}")
        fig.savefig(image_filename, bbox_inches='tight')
        # Sauvegarde des données
        data_filename = f'./data/nappes_data.csv'
        print(f"Saving data to {data_filename}")
        percentage_by_month .to_csv(data_filename, index=False)
        #plt.show()


    def plot_level_pluie(self, spi_result_all_stations):

        spi_result_all_stations['Date'] = pd.to_datetime(spi_result_all_stations['Date'])

        # Ajouter une colonne 'YearMonth' basée sur les colonnes 'Year' et 'Month'
        spi_result_all_stations['YearMonth'] = spi_result_all_stations['Date'].dt.to_period('M')

        # Mapping personnalisé des noms de mois en français
        month_mapping = {
                    "01": "Janvier",
                    "02": "Février",
                    "03": "Mars",
                    "04": "Avril",
                    "05": "Mai",
                    "06": "Juin",
                    "07": "Juillet",
                    "08": "Août",
                    "09": "Septembre",
                    "10": "Octobre",
                    "11": "Novembre",
                    "12": "Décembre"
                }

        # Ignorer les lignes où 'Level' est manquant
        spi_result_all_stations = spi_result_all_stations.dropna(subset=['Level'])

        # Calculer le pourcentage des niveaux par mois
        percentage_by_month = (
            spi_result_all_stations.groupby(['YearMonth', 'Level']).size() /
            spi_result_all_stations.groupby('YearMonth').size() * 100
        )

        # Reformater les données pour le tracé
        percentage_by_month = percentage_by_month.unstack('Level')

        # Vérifier s'il y a treize mois
        if len(percentage_by_month) == 13:
            # Ignorer la première ligne (premier mois)
            percentage_by_month = percentage_by_month.iloc[1:]

        # Tri des colonnes selon l'ordre spécifié
        level_order = ["Sécheresse extrême", "Grande sécheresse", "Sécheresse modérée","Situation normale", "Modérément humide", "Très humide", "Extrêmement humide"]
        percentage_by_month = percentage_by_month[level_order]
        percentage_by_month = percentage_by_month.reset_index().rename(columns={'YearMonth': 'Mois'})
        percentage_by_month['Mois'] = percentage_by_month['Mois'].apply(lambda x: month_mapping[x.strftime('%m')] + ' ' + x.strftime('%Y'))



        # Couleurs
        colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

        # Tracé
        ax = percentage_by_month.plot(kind='bar', stacked=True, color=colors)

        # Ajustements esthétiques
        plt.yticks(range(0, 101, 10))
        plt.title('Pluie: Répartition des niveaux de sécheresse en 2023')
        plt.xlabel('')
        plt.ylabel('Proportion des stations (%)')

        # Utiliser les noms de mois traduits sur l'axe des abscisses
        ax.set_xticklabels(percentage_by_month['Mois'], rotation=45, ha='right')

        # Légende
        legend_labels = ["Sécheresse extrême", "Grande sécheresse", "Sécheresse modérée","Situation normale", "Modérément humide", "Très humide", "Extrêmement humide"]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

        # Réglage automatique de l'orientation des dates sur l'axe des x
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=45)

        # Sauvegarde du graphique
        save_directory="./images/"
        os.makedirs(save_directory, exist_ok=True)
        image_filename = './images/pluviométrie.pdf'
        print(f"Sauvegarde du graphique dans {image_filename}")
        fig.savefig(image_filename, bbox_inches='tight')
        # Sauvegarde des données
        data_filename = f'./data/pluie_data.csv'
        print(f"Saving data to {data_filename}")
        percentage_by_month .to_csv(data_filename, index=False)
        #plt.show()



