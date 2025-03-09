import pandas as pd 
from tester import data

problems_df = {}
for key,value in data.items():
    data = list(map(list, zip(*data)))
    df = pd.DataFrame(data)
    problems_df[key]= df
