import pandas as pd
import matplotlib.pyplot as plt


class DataPrep:

    def __init__(self, data_path='Bike-Sharing-Dataset/hour.csv'):
        self.data_path = data_path

        self.data, self.test_data, self.rides, self.scaled_features = self.clean_data()
        self.target_fields = ['cnt', 'casual', 'registered']
        self.features, self.targets = self.data_loading()

        return

    def pre_vis_data(self):
        rides = pd.read_csv(self.data_path)

        rides[:24 * 10].plot(x='dteday', y='cnt')
        plt.show()
        return

    def clean_data(self):
        rides = pd.read_csv(self.data_path)

        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)

        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        data = rides.drop(fields_to_drop, axis=1)

        # Save data for approximately the last 21 days
        test_data = data[-21 * 24:]

        # Now remove the test data from the data set
        data = data[:-21 * 24]

        # scaling, quantitative features
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

        # Store scalings in a dictionary, so we can convert back later
        scaled_features = {}
        for each in quant_features:
            mean, std = data[each].mean(), data[each].std()
            scaled_features[each] = [mean, std]
            data.loc[:, each] = (data[each] - mean) / std

        return data, test_data, rides, scaled_features

    def data_loading(self):
        # Separate the data into features and targets
        features, targets = self.data.drop(self.target_fields, axis=1), self.data[self.target_fields]

        return features, targets

    def test_loader(self):

        test_features, test_targets = self.test_data.drop(self.target_fields, axis=1), \
                                      self.test_data[self.target_fields]

        return test_features, test_targets

    def train_loader(self):

        train_features, train_targets = self.features[:-60 * 24], self.targets[:-60 * 24]

        return train_features, train_targets

    def val_loader(self):

        val_features, val_targets = self.features[-60 * 24:], self.targets[-60 * 24:]

        return val_features, val_targets

