from plots import plot_tpr_evolution_combined, plot_table

def run_plots(df_setor, df_classification, df_data):


    #############
    # Global #
    ############
    plot_table(df= df_classification, title= 'Tabela de Resultados - Global Classification')
    plot_table(df= df_data, title= "Tabela de Resultados - Data Variation")





    #############
    # Fairness #
    ############
    plot_table(df= df_setor, title= "Tabela de Resultados - Sector")
    plot_tpr_evolution_combined(df= df_setor)

