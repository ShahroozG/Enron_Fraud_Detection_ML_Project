## SH: Create a data frame from the dictionary for EDA
## df is a dataframe equivalant to our data_dict; employee names are index and features are columns of this dataframe
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
print df

### First define a function for converting NaN string as np.nan
def convert_NaN(value):
    if value == "NaN":
        value = np.nan
    else:
        return value
    return value
### applying the function to the whole data frame; So we have np.nan replaced with string NaN
df = df.applymap(convert_NaN)
### Calculating percentage of NaN in each column
print df.isnull().sum()/146.0

### SH: Who are the most NaN holders in the data set
print df.isnull().sum(axis = 'columns').sort_values(ascending=False)