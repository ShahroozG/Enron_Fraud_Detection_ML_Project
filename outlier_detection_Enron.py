### Task 2: Remove outliers
features = ["salary", "bonus"]
### SH: removing the 'TOTAL' outlier
data = featureFormat(data_dict, features)

# SH: plotting data points for having an idea about outliers
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

# SH: finding the outlier with salary more than 25,000,000
for e in data_dict:
    if data_dict[e]['salary'] != "NaN" and data_dict[e]['salary'] > 25000000:
        print e