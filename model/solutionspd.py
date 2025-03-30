import pandas as pd
from solvedict import all_data # Import the combined_data dictionary

solutions_df = {}
for key, value in all_data.items():
    transposed_data = list(map(list, zip(*value)))
    df = pd.DataFrame(transposed_data)
    solutions_df[key] = df

print("Pandas DataFrames for solutions have been created in the 'solutions_df' dictionary.")

# You can now work with the 'solutions_df' dictionary here
# For example, print the DataFrame for a specific key:
# if 'Your Table Heading' in solutions_df:
#     print(solutions_df['Your Table Heading'])