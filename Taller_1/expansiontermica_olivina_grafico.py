from expansiontermicamineral import ExpansionTermicaMineral

olivine=ExpansionTermicaMineral(nombre="Sin nombre", dureza="Sin dureza", rompimiento_por_fractura=None, color="Sin color", composición="Sin composición", lustre="Sin lustre", specific_gravity=0.0, sistema_cristalino="Sin sistema cristalino")
graphite=ExpansionTermicaMineral(nombre="Sin nombre", dureza="Sin dureza", rompimiento_por_fractura=None, color="Sin color", composición="Sin composición", lustre="Sin lustre", specific_gravity=0.0, sistema_cristalino="Sin sistema cristalino")

print(olivine.coef_expansión_termica("olivine_angel_2017.csv"))
print(graphite.coef_expansión_termica("graphite_mceligot_2016.csv"))
