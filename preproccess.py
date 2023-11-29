import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def __del_measurements(data):
    data.drop("torque", axis=1, inplace=True)
    data['mileage'] = data['mileage'].apply(lambda x: float(x[:-5]) if isinstance(x, str) else np.NaN)
    data['engine'] = data['engine'].apply(lambda x: float(x[:-3]) if isinstance(x, str) else np.NaN)
    data['max_power'] = data['max_power'].apply(lambda x: float(x[:-4]) if isinstance(x, str) and len(x) > 4 else np.NaN)

    return data

def __fill_na_cols(data_train, data_test):
    for col in data_train.drop("selling_price", axis=1).columns:
        data_train[f"{col}_isna"] = data_train[col].isna().astype("category")
        data_test[f"{col}_isna"] = data_test[col].isna().astype("category")

        if data_train[col].dtype != "object":
            data_train[col] = data_train[col].fillna(data_train[col].median())
            data_test[col] = data_test[col].fillna(data_test[col].median())
        else:
            data_train[col] = data_train[col].fillna("_")
            data_test[col] = data_test[col].fillna("_")
    
    return data_train, data_test

def __new_features(data_train):
    data_train['max_p_on_engine'] = data_train['max_power']/data_train['engine']

    car_ratio = data_train['year'].max() - (data_train['year'] + 1)
    data_train['km_driven_on_years'] = (data_train['km_driven']/car_ratio).replace(np.inf, 0)

    data_train['in_good_cond'] = (data_train['owner'].isin(["First Owner", "Second Owner"])).astype(int).astype("category")

    features_num = set(data_train.columns[(data_train.dtypes=="number").values]).difference(set(["seats"]))
    for feature in features_num:
        data_train[f"{feature}_sq"] = data_train[feature]**2

    car_brand = data_train['name'].apply(lambda x: x.split(" ")[0])
    data_train["car_brand"] = car_brand
    data_train.drop("name", axis=1, inplace=True)

    return data_train

def __scale_features(data_train, data_test):
    SC = StandardScaler()
    features_num = list(set(data_train.select_dtypes("number").columns).difference(set(["seats"])))
    data_train[features_num] = SC.fit_transform(data_train[features_num])
    data_test[features_num] = SC.transform(data_test[features_num])

    cat_features = data_train.columns[data_train.dtypes=="object"].union(["seats"]).tolist()

    data_train["split_type"] = "train"
    data_test["split_type"] = "test"
    data = pd.concat([data_train, data_test])
    data = pd.get_dummies(data, drop_first=True, columns=cat_features)
    data_train = data[data["split_type"] == "train"]
    data_test = data[data["split_type"] == "test"]
    data_train.drop(columns="split_type", axis=1, inplace=True)
    data_test.drop(columns="split_type", axis=1, inplace=True)
    
    return data_train, data_test

def preproccess_total(data_train, data_test):
    data_train = __del_measurements(data_train)
    data_test = __del_measurements(data_test)
    
    data_train, data_test = __fill_na_cols(data_train, data_test)

    data_train = __new_features(data_train)
    data_test = __new_features(data_test)

    data_train, data_test = __scale_features(data_train, data_test)
    
    model = joblib.load("model.pkl")
    if model.n_features_in_ != data_test.shape[1]:
        features_unset = list(set(model.feature_names_in_) - set(data_test.columns))
        for feature in features_unset:
            data_test[feature] = 0

    return data_test.drop("selling_price", axis=1)