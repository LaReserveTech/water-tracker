from base_standard_index import BaseStandardIndex
from datetime import date, timedelta, datetime
from water_tracker import WaterTracker
from water_tracker import WaterTracker1
def main():
    # Remplacez 'votre_cle_api' par votre clé API réelle
    api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiZWUxMTk3Y2JlMmZlMzY0NGQwOTUyZTZiNzBlZjJlOTU5NzQ4MDFhNWE3MDQ5OTY5OTc0NmNhZjE5MDM5OWFhODJjNWIxMWQxOWQ4NGI5YWQiLCJpYXQiOjE3MDU5OTkwMjYuMjg2NjA0LCJuYmYiOjE3MDU5OTkwMjYuMjg2NjEsImV4cCI6MTcwNzI5NTAyNi4yNzM2NTcsInN1YiI6IjE5MjMwIiwic2NvcGVzIjpbXX0.Cy97gG8bGp-hz9xcVx8GxfRnnk-lVB0bAPhICm9pIgrZUaP3xmGr4yMdmS3H89tCaa88leOMkv0EHoHyn6xYcgFPxI5ZUNvYUpPdpDKi5LwIhs8S5Js6Wc1Dot8J8mvNMcvyQsv1MwQyt0bkaiSEbQD4fsCQn1ic3YvFrsr40r7pclKyLTprbkMAMTO7N9eiyv8AXMbXuVNTEZZpZ2j5d9jCrem07bqAWmsfkXSalL9bRWlXb5uTVRLaGHmE5ReFHjA33dneZYVdiv0iRenctU17WEuhD_GsVgaKkQe80dQOmX5M7ThrVEs-UNokBKOvJzIEm-Bj2OB3ancm3CbpM--WpX47J6aeyhsAGUhWmotgpWn6SVLrriRlDNz_Xj5-8DANudHUG7Hytbzw2PUB413dKTwSTPw_pheXhJt7U65dYT__DwQ-Sk_OkDa_OxVPGOuvryw82ga1RF48dHmyDa0DijQEhhgB5WTqQPd-bfx2jKhGctaCaPvGuDqkfF7joeLEVqGzV9_QAMUROzLKffYvAoxW0A283d7pV90w4j2PxIxrXPK8OVm044Yryvvez_czrS61L_9ScjMKM190qApmqn_D3ReXn7GpkUdE9l09E5gF7gQg4ksncpGETS4LXlNQM35mYKClpDG6BhLBpZqwX-c4_U_N3l-YvoY_vQM"
    
    indicateurs = ["nappes", "pluviométrie"]
    indicator= ["débit"]
 

    # Créez une instance de la classe WaterTracker
    wt = WaterTracker(api_key)
    wt1 = WaterTracker1(api_key)


    # If you haven't built station data and downloaded all timeseries yet uncomment the following lines to do so
    wt.build_stations_data()
    wt.download_all_timeseries()
    wt1.build_stations_data()
    wt1.download_all_timeseries()

    for indicateur in indicateurs:
        # Comment the following line if you don't want to recalculate standardized indicators on each run
        wt.process(indicateur=indicateur)

        # Load existing data
        wt.load()

        # Generate the plot for drought distribution in France
        wt.plot_counts_france(indicateur=indicateur)
    
    for indicateur in indicator:
 
        #wt1.load_timeseries(indicateur=indicateur)


        #wt1.apply_galton_law_to_single_station(station_id)
        
        
        # Appliquer la loi de Galton à toutes les stations
        result_df_all_stations=wt1.apply_galton_law_to_all_stations()
        

        # Créer et afficher le graphique
        wt1.plot_level_percentage(result_df_all_stations)
        
        

if __name__ == "__main__":
    main()
