from water_tracker import WaterTracker

def main():
    # Remplacez 'votre_cle_api'. Il peut arriver que la clé ne marche pas. Dans ce cas,
    #il fraudra se connecter sur https://api.emi.imageau.eu/doc pour actualiser la  clé API.
    api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiZGNmNjgwNzEwODk3NzRiNTIxMjQ2NzQ0ZWRhOTMyMWVjYWMyM2Y1MTlkZTNhNWY3NGEzOWM2NmQwMGY1MWYwNWM3MWJlYTg0YzVmNjQzNjAiLCJpYXQiOjE3MDcyOTU2ODUuMjU3NDQ0LCJuYmYiOjE3MDcyOTU2ODUuMjU3NDQ3LCJleHAiOjE3MDg1OTE2ODUuMjUxMTM5LCJzdWIiOiIxOTIzMCIsInNjb3BlcyI6W119.WOJY9Mz1F9F0o-KwvG3xolwfvK0Rv5s-QSKET6pRA6y7PKFkoVT3_0lYXYZ74xOdRClkULTdpMytCV47qQZG1zfk8AmjLaHBApn4L4NX3en1ndXkNvfjgqcJHhn7BVnGSBbkvUe8qORs_k_zeMWEuZ6TpRYzdLi7HQWHkvErE-RZhXctK82YQSVf-bWgIMx2tFovI-n3LydnfMniDKnLkCRHEpm6x_9ilmHBU_NjagHjf8V1VLI6tvxnORjymy129_1Q1iME8sTJNLiRlZv5ZHR0d2TzK_slXyK-OfhgXKCZbv6ILbjXgyTd5RIVSv_IpuFPTsZd4zWVFYpPrVdsWPT4-5k_KcUo1X8UzqQYG__MXFxqRdUBf0z8q7xOkmkulFgTovz2PGXQwmkWGwHf6M_wsjzKuLyqzYXIRC_n1wbHoQ6Q25HCn3P_0clZIUlLkj_thsBnZLh3R09va7XbtHWrtS_xYWeoTRToea8EE9hL1Yp-wSs_Y8lLWJjnDA_bfN_qjaJUy86eCQl3v6WPGcCiB9ezlvNg1_8Ix1uSGasgBy3UY7YNM3_QcgAoKPp5xnnQNcGi3eRXhYqTpc1qNJpIka0f69rblm4_3op2iXbKBm_7ZvYoJfcE6C_UQA6GnZfmLeHRTdotmTG1li89FuyELGAmxR-VUjz7y8oX-K0"
    
    # Créez une instance de la classe WaterTracker
    wt = WaterTracker(api_key)
    # téléchargez les données
    wt.build_stations_data()
    wt.download_all_timeseries()
    # Calculez les indicateurs nappes, pluie et débit
    sgi_result_all_stations = wt.sgi_to_all_stations() #le code est fait pour
    spi_result_all_stations = wt.spi_to_all_stations()
    result_df_all_stations = wt.apply_galton_law_to_all_stations()
    # Affichez le graphique des indicateurs nappes, pluie et débit
    wt.plot_level_nappes(sgi_result_all_stations)
    wt.plot_level_pluie(spi_result_all_stations)
    wt.plot_level_debit(result_df_all_stations)
 
        

if __name__ == "__main__":
    main()
