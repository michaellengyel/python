import math

import pandas as pd
import numpy as np


def basics():
    print("pandas basics")

    df = pd.read_csv("data/personal.csv")

    print("Accessing the shape of the data frame")
    print(df.shape)

    print("Printing the column names (labels):")
    print(df.columns)

    print("Accessing a column (access it like a key):")
    print(df.income)

    print("Alternative access to a column (access it like a key):")
    print(df["income"])

    print("Accessing multiple columns (returns a data frame):")
    print(df[["income", "level"]])

    print("Accessing a row by integer (iloc):")
    print(df.iloc[0])

    print("Accessing multiple rows by integer:")
    print(df.iloc[[0, 1]])

    print("Accessing an element in a row by integer:")
    print(df.iloc[0, 2])

    print("Accessing an element in a row by label:")
    print(df.loc[[8], ["state"]])

    print("Accessing an column section in a row by label:")
    print(df.loc[[3, 6], ["name"]])

    print("Clustering/sorting of a column's data:")
    print(df["class"].value_counts())

    print("Accessing a row by label (loc)")
    print(df.loc[2])

    print("Accessing an element of multiple rows:")
    print(df.loc[[2, 4, 5], "income"])

    print("Accessing a slice of elements in a column:")
    print(df.loc[2:5, "class"])

    print("Accessing a 2D slice of data:")
    print(df.loc[3:7, "date":"class"])

    """
    Sometimes it makes sense to set a specific label to be the index of the data frame, e.g. email address or
    phone number. This is because you might want the data may not have a unique id label.
    """
    print("Changing the default index to another label:")
    df.set_index("email", inplace=True)
    print(df.index)
    print("Print using custom chosen index")
    print(df.loc["rick@rick.rick"])
    # print(df.loc[3]) # <- this would throw error
    print(df.iloc[8])  # <- this still works
    df.reset_index(inplace=True)
    print("Print using default index")
    print(df.loc[3])


def intermediate():
    print("pandas intermediate")

    print("Changing the default index column during data read:")
    df = pd.read_csv("data/personal.csv", index_col="email")
    print(df.head)

    print("Filtering data: Set the filtered value of each row to true/false based on condition")
    print(df['class'] == 'fish')

    print("Filtering data: If condition is true, add it to a list or frame")
    fish_filter = df['class'] == 'fish'
    print(df[fish_filter])
    print("...Alternatively:")
    print(df.loc[fish_filter])
    print("Get an attribute of a filtered frame:")
    print(df.loc[fish_filter, 'name'])
    # print(df[df['class'] == 'fish']) <- or just one line

    print("Multiple filters:")
    fish_filter = df['class'] == 'fish'
    name_filter = df['name'] == 'bob'
    # OR
    print(df.loc[fish_filter | name_filter])
    # AND
    print(df.loc[fish_filter & name_filter])

    print("Filter for id range:")
    min_id_filter = df['id'] > 5
    max_id_filter = df['id'] < 9
    print(df.loc[min_id_filter & max_id_filter, ['id', 'income', 'class']])

    print("Filtering for specific options:")
    names = ['rich', 'jerry', 'mark', 'hodor']
    names_filter = df['name'].isin(names)
    print(df.loc[names_filter])
    print(df.loc[names_filter, 'income'])

    print("Renaming columns:")
    df.columns = ['email_address', 'identification', 'date', 'income', 'class', 'level', 'state']
    print(df)
    # Or using list comprehension:
    df.columns = [x.upper() for x in df.columns]
    print(df)
    print("Rename a single column name:")
    df.rename(columns={'IDENTIFICATION': 'id'}, inplace=True)
    print(df)

    # Data Reset:

    df = pd.read_csv("data/personal.csv")
    print('Updating data in our rows:')

    # Change some of the data:
    print(df.iloc[2])
    df.loc[2, ['income', 'class']] = ['99999', 'orangutan']
    print(df.loc[2])

    # Change a single element:
    print(df.iloc[3])
    df.loc[3, 'income'] = 42
    print(df.iloc[3])

    print(df.iloc[3])
    df.at[3, 'income'] = 42000
    print(df.iloc[3])

    # apply (applies a function on values)
    # map ()
    # applymap
    # replace

    # *** APPLY ***

    print("Apply len() function to each element of a series")
    # Note: df['email'] accesses a series
    print(df['email'].apply(len))

    print("Apply custom function of each element of a column")

    def switch_state(state):
        m_state = 0
        if state == 0:
            m_state = 1
        elif state == 1:
            m_state = 0
        else:
            print("Error: Non binary value found!")
        return m_state

    print(df['state'].apply(switch_state))

    # To actually change the values:
    df['state'] = df['state'].apply(switch_state)
    print(df)

    # Passing lambda functions:
    df['name'] = df['name'].apply(lambda x: x.upper())
    print(df)

    print("Applying a function on each row of a data frame")
    print(df.apply(len, axis='columns'))  # axis='rows' is the default

    print("Applying a function on a data frame")
    # print(df.apply(np.sqrt)) <- only works with numeric data

    # *** APPLYMAP ***

    print("applymap to take srqt of data frame elements:")
    df = pd.read_csv("data/numbers.csv")
    print(df.applymap(np.sqrt))

    print("applymap to use custom function:")

    def nan_to_num(number):
        if math.isnan(number):
            return 0
        else:
            return number

    print(df.applymap(nan_to_num))

    # *** MAP ***

    # Problem with map is that is makes all unchanged values into nan-s
    print("Using map to change values:")
    print(df['4'].map({4: 9999}))

    # *** REPLACE ***

    print("Using map to change values:")
    print(df['4'].replace({4: 9999}))

    # ADDING AND REMOVING COLUMNS AND ROWS

    df = pd.read_csv("data/personal.csv")

    print("Adding and removing columns:")
    df['title'] = df['class'] + ' ' + df['name']
    print(df)

    df.drop(columns=['state', 'income'], inplace=True)
    print(df)

    print("Splitting columns or adding columns:")
    df[['address', 'provider']] = df['email'].str.split('@', expand=True)
    print(df)

    print("Adding a row")
    df = df.append({'name': 'henry'}, ignore_index=True)
    print(df)

    print('Removing a row:')
    df.drop(index=2, inplace=True)
    print(df)

    print("Filtering a row based on a condition:")
    df.drop(index=df[df['name'] == 'rich'].index, inplace=True)
    print(df)

    print("Filtering a row based on a condition:")
    name_filter = df['name'] == 'maul'
    df.drop(index=df[name_filter].index, inplace=True)
    print(df)


def advanced():
    print("pandas advanced:")

    df = pd.read_csv("data/personal.csv")

    print("Sorting data frames alphabetically:")
    print(df.sort_values(by='name'))

    print("Sorting data frames reverse alphabetically:")
    print(df.sort_values(by='name', ascending=False))

    print("Multiple sorting criteria, e.g. bz class first then by name:")
    print(df.sort_values(by=['class', 'name'], ascending=True))

    print("Multiple sorting criteria for both data and sorting order by using lists:")
    print(df.sort_values(by=['class', 'name'], ascending=[False, True]))

    print("Sorting based on index:")
    print(df.sort_index())

    print("Sorting just a column:")
    print(df['date'].sort_values())

    print("Get/Sort the 3 largest incomes:")
    print(df['income'].nlargest(3))

    print("Get/Sort the 3 largest income's rows:")
    print(df.nlargest(3, 'income'))

    print("Get/Sort the 3 smallest income's rows:")
    print(df.nsmallest(3, 'income'))

    # *** Aggregate data ***

    print("Aggregate data section:")

    print("Get the median of a column:")
    print(df['income'].median())

    print("Using describe() to give a overview of data:")
    print(df.describe())

    print("Count the occurrence of a value")
    print(df['state'].value_counts())

    print("Count the occurrence of a value and then normalize")
    print(df['state'].value_counts(normalize=True))

    print("Group by operation: Splitting the object, applying a function, combining the results")
    print(df['class'].value_counts())

    class_group = df.groupby('class')  # Create groups based on different classes
    print(class_group.get_group('fish'))  # Get the fish class group

    print("Count the occupancies of income by class based groups")
    print(class_group['income'].value_counts())

    print("Count the occupancies of income by class based groups, but only the first 3")
    print(class_group['income'].value_counts().head(3))

    # print("Alternatively, use a filter:")
    # fish_filter = df['class'] == 'fish'
    # print(df.loc[fish_filter]['name'].value_counts())

    df = pd.read_csv("data/numbers.csv")

    print("Grouping of numeric data")

    group_seven = df.groupby('7')
    print(group_seven['3'].value_counts(normalize=True))

    print("")
    print(group_seven['3'].mean())

    print("Cleaning Data:")

    print("Dropping rows and columns with missing values:")
    print(df.dropna())

    df = pd.read_csv("data/numbers.csv")
    df.replace(math.nan, np.nan, inplace=True)
    print(df)

    print("Show data types of columns:")
    print(df.dtypes)

    df['1'] = df['1'].astype(float)
    print(df.dtypes)

    df = df.astype(float)
    print(df.dtypes)

    # *** Writing to a file ***
    print("Writing to a file:")
    df = pd.read_csv("data/numbers.csv", delimiter=',')
    df.replace(np.nan, '0', inplace=True)
    df.to_csv("data/csv_numbers.csv")
    df.to_csv("data/tsv_numbers.tsv", sep='\t')


def main():
    # basics()
    # intermediate()
    advanced()


if __name__ == '__main__':
    main()
