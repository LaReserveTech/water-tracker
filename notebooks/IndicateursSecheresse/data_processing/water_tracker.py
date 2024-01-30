from base_standard_index import BaseStandardIndex
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import requests
from tqdm import tqdm
from scipy.stats import lognorm

class WaterTracker():
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_folder = "data/"
        #self.timeseries_folder = self.data_folder+"timeseries/"
        #self.df_stations = pd.read_csv(f"{self.data_folder}df_stations.csv")
        self.mapping_indicateur_column = {"pluviométrie":"dryness-meteo", 
                                          "nappes": "dryness-groundwater"
                                          }
        self.mapping_indicateur_indicateur_standardise = {"pluviométrie":"spi", 
                                                          "nappes": "spli"
                                                          }
        self.levels_colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]
    
        self.timeseries = {}
        self.timeseries_computed = {}
        self.standardized_indicator_means_last_year = {}
        self.aggregated_standardized_indicator_means_last_year = {}
        self.levels_pluviometrie = {
            -2.00: "Sécheresse extrême",
            -1.50: "Grande sécheresse",
            -1.00: "Sécheresse modérée",
            1.00: "Situation normale",
            1.50: "Modérément humide",
            2.00: "Très humide",
            float('inf'): "Extrêmement humide"
        }
        self.levels_nappes = {
            -1.28: "Très bas",
            -0.84: "Bas",
            -0.25: "Modérément bas",
            0.25: "Autour de la normale",
            0.84: "Modérément haut",
            1.28: "Haut",
            float('inf'): "Très haut"
        }
        self.levels = None
        self.mapping_indicator_names = {"dryness-meteo":"rain-level",
                                        "dryness-groundwater":"water-level-static"
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
        for i in tqdm(range(1,96)):
            stations_data[i] = self.download_departement_data(str(i).rjust(2,"0"))
        # data[20] is None
        del stations_data[20]
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
        """
        Call this function to query the API and download/update the timeseries for all stations.
        The file df_stations.csv must be created first with self.build_stations_data()
        """
        start_date = "1970-01-01"
        today = datetime.today().strftime('%Y-%m-%d')
        df = pd.read_csv("./data/df_stations.csv")
        cpt = 0
        n = len(df)
        os.makedirs(f"{os.path.dirname(__file__)}/data/timeseries", exist_ok=True)
        for i, row in df.iterrows():
            station_id = row["id"]
            if station_id not in self.black_listed_station_ids:
                cpt+=1
                print(f"{100*cpt/n}%", end="\r")
                if row["dryness-meteo"]==1 or row["dryness-groundwater"]==1:

                    filename = f"./data/timeseries/{station_id}.csv"
                    d = None 
                    if not os.path.isfile(filename):
                        d = self.download_timeseries_station(station_id, start_date, today)
                    if d is not None:  # Check if the download is successful    
                        if row["dryness-meteo"]==1:
                            timeseries = pd.DataFrame(d[self.mapping_indicator_names["dryness-meteo"]])[["date","value"]]
                        elif row["dryness-groundwater"]==1:
                            timeseries = pd.DataFrame(d[self.mapping_indicator_names["dryness-groundwater"]])[["date","value"]]
                        else:
                            pass
                        timeseries["date"] = pd.to_datetime(timeseries["date"])
                        timeseries = timeseries.drop_duplicates()
                        #timeseries = timeseries.set_index("date")
                        timeseries.to_csv(filename)

                    else:
                        timeseries_init = pd.read_csv(filename)
                        timeseries_init["date"] = pd.to_datetime(timeseries_init["date"])
                        # Download data from the last date only
                        #next_date = timeseries_init.index[0].to_pydatetime()+timedelta(days=1)
                        next_date = (timeseries_init["date"].max().to_pydatetime()+timedelta(days=1)).date()
                        #print(loc_id, next_date, today)
                        d = self.download_timeseries_station(station_id, next_date, today)
                        #print(d)
                        if row["dryness-meteo"]==1:
                            indicator_name = self.mapping_indicator_names["dryness-meteo"]
                        elif row["dryness-groundwater"]==1:
                            indicator_name = self.mapping_indicator_names["dryness-groundwater"]
                        else:
                            print("ERREUR")

                        if len(d[indicator_name]) > 0:
                            timeseries = pd.DataFrame(d[indicator_name])[["date","value"]]
                            timeseries["date"] = pd.to_datetime(timeseries["date"])
                            timeseries = timeseries.drop_duplicates()
                            #timeseries = timeseries.set_index("date")
                            timeseries_final = pd.concat([timeseries_init, timeseries], axis=0)
                            timeseries_final.to_csv(filename)


    def column_from_indicateur(self, indicateur):
        if indicateur in self.mapping_indicateur_column.keys():
            return self.mapping_indicateur_column[indicateur]
        else:
            print("Problème: l'indicateur doit être pluviométrie ou nappes")
            return None

    def load_timeseries(self):
        self.timeseries = pickle.load(open(f"{self.data_folder}timeseries.pkl", "rb"))


    def load_timeseries_computed(self):
        self.timeseries_computed = pickle.load(open(f"{self.data_folder}timeseries_computed.pkl", "rb"))


    def load_standardized_indicator_means_last_year(self):
        self.standardized_indicator_means_last_year = pickle.load(open(f"{self.data_folder}standardized_indicator_means_last_year.pkl", "rb"))


    def load_aggregated_standardized_indicator_means_last_year(self):
        self.aggregated_standardized_indicator_means_last_year = pickle.load(open(f"{self.data_folder}aggregated_standardized_indicator_means_last_year.pkl", "rb"))

    def load_timeseries_from_files(self, indicateur, min_number_years=15):
        column = self.column_from_indicateur(indicateur)
        if column is None:
            return None

    
        ids = self.df_stations[self.df_stations[column]==1]["id"].values
        self.timeseries[indicateur] = {}
        print(f"Chargement des chroniques pour l'indicateur {indicateur}")
        for station_id in tqdm(ids) :
            if station_id not in self.black_listed_station_ids:
                timeseries = pd.read_csv(f"{self.timeseries_folder}{station_id}.csv")
                timeseries["date"] = pd.to_datetime(timeseries["date"])

                if (timeseries["date"].max() - timeseries["date"].min()).days/365 >= (min_number_years+1):
                    start_date = (date.today()-timedelta(days=min_number_years*365)).strftime("%Y-%m-%d")
                    timeseries = timeseries[timeseries["date"]>=start_date]
                    timeseries = timeseries.set_index("date")
                    self.timeseries[indicateur][station_id] =timeseries
                    pickle.dump(self.timeseries, open(f"{self.data_folder}timeseries.pkl", "wb"))     
        print(f"Terminé")

  
    def save_data(self, data, filename):
        print(f"Saving into {filename}")
        pickle.dump(data, open(filename, "wb"))

    
    def save_timeseries(self):
        print(f"Saving timeseries into {self.data_folder}timeseries.pkl")
        pickle.dump(self.timeseries, open(f"{self.data_folder}timeseries.pkl", "wb"))

    

            

    def save_timeseries_computed(self):
        print(f"Saving timeseries_computed into {self.data_folder}timeseries_computed.pkl")
        pickle.dump(self.timeseries_computed, open(f"{self.data_folder}timeseries_computed.pkl", "wb"))

    
    def save_standardized_indicator_means_last_year(self):
        print(f"Saving standardized_indicator_means_last_year into {self.data_folder}standardized_indicator_means_last_year.pkl")
        pickle.dump(self.standardized_indicator_means_last_year, open(f"{self.data_folder}standardized_indicator_means_last_year.pkl", "wb"))



    def save_aggregated_standardized_indicator_means_last_year(self):
        print(f"Saving aggregated_standardized_indicator_means_last_year into {self.data_folder}aggregated_standardized_indicator_means_last_year.pkl")
        pickle.dump(self.aggregated_standardized_indicator_means_last_year, open(f"{self.data_folder}aggregated_standardized_indicator_means_last_year.pkl", "wb"))

    

    def save(self):
        """
        Save all the data that are stored
        """
        self.save_timeseries()
        self.save_timeseries_computed()
        self.save_standardized_indicator_means_last_year()
        self.save_aggregated_standardized_indicator_means_last_year()

        

    def load(self):
        """
        Loads all the data that are stored
        """
        print(f"Chargement des chroniques (timeseries) depuis {self.data_folder}timeseries.pkl")
        self.timeseries = pickle.load(open(f"{self.data_folder}timeseries.pkl", "rb"))

        print(f"Chargement des chroniques des indicateurs (timeseries_computed) depuis {self.data_folder}timeseries_computed.pkl")
        self.timeseries_computed = pickle.load(open(f"{self.data_folder}timeseries_computed.pkl", "rb"))

        print(f"Chargement des indicateurs standardisés sur 1 an (standardized_indicator_means_last_year) depuis {self.data_folder}standardized_indicator_means_last_year.pkl")
        self.standardized_indicator_means_last_year = pickle.load(open(f"{self.data_folder}standardized_indicator_means_last_year.pkl", "rb"))

        print(f"Chargement des données agrégées (aggregated_standardized_indicator_means_last_year) depuis {self.data_folder}aggregated_standardized_indicator_means_last_year.pkl")
        self.aggregated_standardized_indicator_means_last_year = pickle.load(open(f"{self.data_folder}aggregated_standardized_indicator_means_last_year.pkl", "rb"))
    def aggregate_standardized_indicator_means_last_year(self, indicateur):
        if indicateur == "nappes":
            self.levels = self.levels_nappes
        else:
            self.levels = self.levels_pluviometrie
        self.aggregated_standardized_indicator_means_last_year[indicateur] = { month: {level:0  for level in range(len(self.levels.keys()))} for month in range(12)}
        print(self.aggregated_standardized_indicator_means_last_year)
        for station_id, data in self.standardized_indicator_means_last_year[indicateur].items():
            if station_id not in self.black_listed_station_ids:
                for month, level in data.items():
                    self.aggregated_standardized_indicator_means_last_year[indicateur][month][level] += 1



    def compute_standardized_indicator_values(self, indicateur, freq="M", scale = 1):
        print("Calcul des indicateurs standardisés par mois")
        self.standardized_indicator_means_last_year[indicateur] = {}
        self.timeseries_computed[indicateur] = {}

        standardized_indicator = self.mapping_indicateur_indicateur_standardise[indicateur]
        end_date = date.today().replace(day=1)
        one_year_before = (end_date.today()-timedelta(days=365)).strftime("%Y-%m-%d")

        for station_id, timeseries in tqdm(self.timeseries[indicateur].items()):
            if station_id not in self.black_listed_station_ids:
                timeseries = self.clean_timeseries(timeseries)

                standardized_indicator_computer = BaseStandardIndex()
                timeseries_computed = standardized_indicator_computer.calculate(df=timeseries,
                                                                                date_col='date', 
                                                                                precip_cols='value',
                                                                                indicator=standardized_indicator, 
                                                                                freq=freq, 
                                                                                scale=scale, # rolling sum over 1 month
                                                                                fit_type="mle", 
                                                                                dist_type="gam",
                                                                                )

                

                timeseries_computed.columns = ["date", f"roll_{scale}{freq}", standardized_indicator]

                self.timeseries_computed[indicateur][station_id] = timeseries_computed

                timeseries_tmp = timeseries_computed[timeseries_computed["date"] >= one_year_before]

                d = dict(timeseries_tmp[standardized_indicator].groupby(timeseries_tmp['date'].dt.month).mean().apply(lambda x: self.standardized_indicator_to_level_code(x,indicateur)))
                dd = {k-1:v for k,v in d.items()}
                self.standardized_indicator_means_last_year[indicateur][station_id] = dd


    def plot_counts_france(self, indicateur):
        """
        Plots the BRGM representation of the proportions of stations in France for each dryness levels month by month since 1 year.
        Returns also the corresponding dataframe
        """
        print(f"Création du graphique pour {indicateur}")
        
        if indicateur == "nappes":
            self.levels = self.levels_nappes
        else:
            self.levels = self.levels_pluviometrie
        standardized_indicator = self.mapping_indicateur_indicateur_standardise[indicateur]
        df_levels = pd.DataFrame(self.aggregated_standardized_indicator_means_last_year[indicateur]).transpose().reset_index()
        df_levels.columns = ["Mois"] + [x for x in self.levels.values()]
        dict_months ={  0: "Janvier",
                        1: "Février",
                        2: "Mars",
                        3: "Avril",
                        4: "Mai",
                        5: "Juin",
                        6: "Juillet",
                        7: "Août",
                        8: "Septembre",
                        9: "Octobre",
                        10: "Novembre",
                        11: "Décembre",}

        # Add years after months in dict_months
        today = date.today()
        current_year = today.year
        last_year = current_year-1
        last_month = today.month-1
        for month, name in dict_months.items():
            if month<last_month:
                dict_months[month] = f"{name} {current_year}"
            else:
                dict_months[month] = f"{name} {last_year}"

        df_levels["Mois"]= df_levels["Mois"].replace(dict_months)

        # Shift rows to display the last month on the right of the graph
        df_levels = df_levels.reindex(index=np.roll(df_levels.index,12-(date.today().replace(day=1).month-1)))
        df_values = df_levels.drop("Mois", axis=1)
        df_values = df_values.div(df_values.sum(axis=1), axis=0)*100
        df_levels = pd.concat([df_levels["Mois"], df_values], axis=1)

        ax = df_levels.plot.bar(x="Mois",
                                stacked=True,
                                title=f"{indicateur.capitalize()} : répartition de la sécheresse depuis 1 an",
                                color=self.levels_colors,
                                grid=False,
                                )
        plt.yticks(range(0,101,10))
        plt.xlabel("")
        plt.ylabel("Proportion des stations (%)")

        #plt.tick_params(labelright=True)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left',bbox_to_anchor=(1.0, 0.5))
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=45)
        image_filename = f'./images/{indicateur}.pdf'
        print(f"Sauvegarde du graphique dans {image_filename}")
        fig.savefig(image_filename, bbox_inches='tight')

        # The following code is commented out and seems to be a duplicate
        data_filename = f'./data/{indicateur}_data.csv'
        print(f"Saving data to {data_filename}")
        df_levels.to_csv(data_filename, index=False)

        return df_levels

    def standardized_indicator_to_level_code(self, standardized_indicator_value,indicateur):
        if indicateur == "nappes":
           self.levels = self.levels_nappes
        else:
           self.levels = self.levels_pluviometrie

        for i, (k, v) in enumerate(self.levels.items()):
            if standardized_indicator_value < k:
                return i




    



    def process(self, indicateur):
        """
        Loads the timeseries
        indicateur = "pluviométrie", "nappe", "nappe profonde"
        """
        self.load_timeseries_from_files(indicateur=indicateur, min_number_years=15)
        self.compute_standardized_indicator_values(indicateur=indicateur, freq="M", scale=1)
        self.save_timeseries_computed()
        self.save_standardized_indicator_means_last_year()
        self.aggregate_standardized_indicator_means_last_year(indicateur=indicateur)
        self.save_aggregated_standardized_indicator_means_last_year()


    def test(self, indicateur, id_station):
        freq="M"
        scale = 1
        end_date = date.today().replace(day=1)
        one_year_before = (end_date.today()-timedelta(days=365)).strftime("%Y-%m-%d")
        timeseries = pd.read_csv(f"./data/timeseries/{id_station}.csv")#self.timeseries[indicateur][id_station]
        timeseries = self.clean_timeseries(timeseries)
        standardized_indicator = self.mapping_indicateur_indicateur_standardise[indicateur]

        timeseries = timeseries.reset_index()
        timeseries = timeseries.drop_duplicates(subset="date")
        #print(timeseries["value"].tolist())
        print(timeseries["value"].describe())
        standardized_indicator_computer = BaseStandardIndex()
        timeseries_computed = standardized_indicator_computer.calculate(df=timeseries,
                                                                        date_col='date', 
                                                                        precip_cols='value',
                                                                        indicator=standardized_indicator, 
                                                                        freq=freq, 
                                                                        scale=scale, # rolling sum over 3 month
                                                                        fit_type="mle", 
                                                                        dist_type="gam",
                                                                        )

        

        #if indicateur == "pluviométrie":
                    # Utilisation de la somme glissante sur 3 mois
                    #timeseries_computed.columns = ["date", f"roll_{scale}{freq}", standardized_indicator]

       # else:
                    # Utilisation de la moyenne glissante sur 3 mois
                    #timeseries_computed.columns = ["date", f"mean_{scale}{freq}", standardized_indicator]
        #self.timeseries_computed[indicateur][id_station] = timeseries_computed

        timeseries_computed.columns = ["date", f"roll_{scale}{freq}", standardized_indicator]
        timeseries_tmp = timeseries_computed[timeseries_computed["date"] >= one_year_before]
        #print(timeseries["value_scale_3"].tolist())
        #print(list(timeseries_computed["spli"].values))
        print(timeseries_computed.describe())
        d = dict(timeseries_tmp[standardized_indicator].groupby(timeseries_tmp['date'].dt.month).mean().apply(lambda x: self.standardized_indicator_to_level_code(x,indicateur)))
        dd = {k-1:v for k,v in d.items()}
        print(dd)
        #self.standardized_indicator_means_last_year[indicateur][id_station] = dd



    def clean_timeseries(self, timeseries):
        """
        Remove outliers, duplicated dates and set all values to positive values
        """
        df = timeseries.copy()
        df = df.reset_index()
        df = df.drop_duplicates(subset="date")
        df["value"] = df["value"].apply(abs)
        col="value"
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3- Q1
        c = 2
        min_t = Q1 - c*IQR
        max_t = Q3 + c*IQR
        df["outlier"] = (df[col].clip(lower = min_t,upper=max_t) != df[col])
        return df[~df["outlier"]].drop("outlier",axis=1)



class WaterTracker1():
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_folder = "data2/"
        #self.load_timeseries ={}
        self.mapping_indicateur_column ={"débit": "dryness-stream-flow"}
        self.timeseries = {}
        self.levels = None
        self.mapping_indicator_names = {"dryness-stream-flow": "stream-flow"}

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
        stations_data = self.download_stations_data()
        res = []
        for departement_code in stations_data.keys():
            for station in stations_data[departement_code]["data"]["locations"]:
                data = {"departement_code": departement_code, "id": station["id"],
                        'name': station["name"], 'bss_code': station["bss_code"],
                        "indicators": [indicator["state"]["type"]["name"] for indicator in station["indicators"]]}
                res.append(data)

        data_folder = os.path.abspath("./data2")
        os.makedirs(data_folder, exist_ok=True)

        df_stations = pd.DataFrame(res)
        df_stations = pd.concat([df_stations, pd.get_dummies(df_stations["indicators"].explode()).groupby(level=0).sum()],
                                axis=1).drop("indicators", axis=1)
        output_filename = os.path.abspath("./data2/df_stations.csv")
        print(f"Sauvegarde des données des stations dans {output_filename}")
        df_stations.to_csv(output_filename, index=False)
        self.timeseries_folder = self.data_folder + "timeseries/"
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
                else:
                    continue

                filename = os.path.join(self.timeseries_folder, f"{station_id}.csv")
                d = None

                if not os.path.isfile(filename):
                    # Téléchargez toutes les données disponibles pour chaque station
                    d = self.download_timeseries_station(station_id, "1970-01-01", today)

                    # Vérifiez si au moins 15 ans de données sont disponibles
                    if d and indicator_name in d and len(d[indicator_name]) >= 10 * 365:
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
                        print(f"La station {station_id} n'a pas suffisamment de données sur 10 ans. Les données ne seront pas téléchargées.")
                        continue

               
                timeseries_init = pd.read_csv(filename)
                timeseries_init["date"] = pd.to_datetime(timeseries_init["date"])
                next_date = (timeseries_init["date"].max().to_pydatetime() + timedelta(days=1)).date()
                d = self.download_timeseries_station(station_id, next_date, today)

                if indicator_name in d:
                    if len(d[indicator_name]) > 0:
                        timeseries = pd.DataFrame(d[indicator_name])[["date", "value"]]
                        timeseries["date"] = pd.to_datetime(timeseries["date"])
                        timeseries = timeseries.drop_duplicates()
                        timeseries_final = pd.concat([timeseries_init, timeseries], axis=0)
                        timeseries_final.to_csv(filename)



    def column_from_indicateur(self, indicateur):
        if indicateur in self.mapping_indicateur_column.keys():
            return self.mapping_indicateur_column[indicateur]
        else:
            print("Problème: l'indicateur doit être débit")
            return None

    def load_timeseries(self, indicateur):
        data_folder = os.path.abspath("./data2")
        self.timeseries_folder = self.data_folder + "timeseries/"
        column = self.column_from_indicateur(indicateur)
        if column is None:
            return None
        
        self.df_stations = pd.read_csv(f"{self.data_folder}df_stations.csv")
        ids = self.df_stations[self.df_stations[column] == 1]["id"].values

        self.timeseries[indicateur] = {}
        print(f"Chargement des chroniques pour l'indicateur {indicateur}")

        for station_id in tqdm(ids):
            if station_id not in self.black_listed_station_ids:
                filename = os.path.join(self.timeseries_folder, f"{station_id}.csv")

                # Vérifier si le fichier existe avant de le charger
                if os.path.isfile(filename):
                    timeseries = pd.read_csv(filename)
                    timeseries["date"] = pd.to_datetime(timeseries["date"])
                    timeseries = timeseries.set_index("date")
                    self.timeseries[indicateur][station_id] = timeseries

        print("Terminé")




    def process(self, indicateur):
        """
        Loads the timeseries
        indicateur = "débit", "nappes", "température"
        """
        self.load_timeseries(indicateur=indicateur)



    def apply_galton_law_to_single_station(self, station_id):
        # Vérifier si le fichier station_id.csv existe dans le dossier timeseries
        filename = f"./data2/timeseries/{station_id}.csv"
        if os.path.isfile(filename):
            timeseries = pd.read_csv(filename)

            # Assurez-vous que la colonne "date" est de type datetime
            timeseries["date"] = pd.to_datetime(timeseries["date"])

            # Assurez-vous que la colonne "value" est numérique
            timeseries["value"] = pd.to_numeric(timeseries["value"], errors='coerce')

            # Vérifiez si la colonne "value" est présente et a suffisamment de données
            if "value" in timeseries.columns and len(timeseries) >= 10 * 365:
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
                one_year_ago = result_df['Date'].max() - pd.DateOffset(months=12)


                # Filtrer les données pour les douze derniers mois
                result_df = result_df[result_df['Date'] >= one_year_ago]


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
        # Créer un DataFrame avec toutes les dates pour la première et la dernière date dans le DataFrame
        all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq='D')

        # Réinitialiser l'index du DataFrame complet
        all_dates_df = pd.DataFrame({"Date": all_dates})
        all_dates_df.reset_index(drop=True, inplace=True)

        # Fusionner le DataFrame complet avec le DataFrame original
        result_df = pd.merge(all_dates_df, df, on="Date", how="left")

        return result_df



    def apply_galton_law_to_all_stations(self):
        data_folder = "./data2/timeseries/"
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

        # Trier le DataFrame global par année puis par mois
        result_df_all_stations.sort_values(by="Date", inplace=True)

        # Réinitialiser les index
        result_df_all_stations.reset_index(drop=True, inplace=True)
        # Compléter les dates manquantes
        result_df_all_stations = self.complete_missing_months(result_df_all_stations)
        # Sélectionner la date du jour moins un an en arrière
        one_year_ago = result_df_all_stations['Date'].max() - pd.DateOffset(months=12)

        # Filtrer les données pour les douze derniers mois
        result_df_all_stations = result_df_all_stations[result_df_all_stations['Date'] >= one_year_ago]


        # Sauvegarder le DataFrame résultant dans un fichier pickle
        #pickle_filename = "./data2/galton_results/result_df_all_stations.pkl"
        #result_df_all_stations.to_pickle(pickle_filename)
        #print(f"DataFrame sauvegardé dans {pickle_filename}")

        return result_df_all_stations





    def plot_level_percentage(self, result_df_all_stations):

        # Convertir la colonne 'Date' en datetime
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

        # Couleurs
        colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

        # Tracé
        ax = percentage_by_month.plot(kind='bar', stacked=True, color=colors)

        # Ajustements esthétiques
        plt.yticks(range(0, 101, 10))
        plt.title('Répartition des niveaux de sécheresse au cours de la dernière année (VCN3)')
        plt.xlabel('')
        plt.ylabel('Proportion des stations (%)')

        # Utiliser les noms de mois traduits sur l'axe des abscisses
        ax.set_xticklabels([month_mapping[date.strftime('%m')] +" "+date.strftime('%Y') for date in percentage_by_month.index], rotation=45, ha='right')

        # Légende
        legend_labels = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

        # Réglage automatique de l'orientation des dates sur l'axe des x
        fig = ax.get_figure()
        fig.autofmt_xdate(rotation=45)

        # Sauvegarde du graphique
        image_filename = './images/debit_repartition_secheresse.pdf'
        print(f"Sauvegarde du graphique dans {image_filename}")
        fig.savefig(image_filename, bbox_inches='tight')
        # Sauvegarde des données
        data_filename = f'./data/débit_data.csv'
        print(f"Saving data to {data_filename}")
        percentage_by_month .to_csv(data_filename, index=False)
        plt.show()


