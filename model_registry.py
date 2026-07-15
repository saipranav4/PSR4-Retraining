from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import (

    RandomForestClassifier, RandomForestRegressor,

    GradientBoostingClassifier, GradientBoostingRegressor,

    AdaBoostClassifier, AdaBoostRegressor,

    ExtraTreesClassifier, ExtraTreesRegressor

)

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier, XGBRegressor

from lightgbm import LGBMClassifier, LGBMRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
 
MODEL_REGISTRY = {

    "classification": {

        "LogisticRegression": LogisticRegression,

        "RandomForestClassifier": RandomForestClassifier,

        # "GradientBoostingClassifier": GradientBoostingClassifier,

       # "AdaBoostClassifier": AdaBoostClassifier,

       # "ExtraTreesClassifier": ExtraTreesClassifier,

        "XGBClassifier": XGBClassifier,

        "LGBMClassifier": LGBMClassifier,

        "CatBoostClassifier": CatBoostClassifier,

        # "SVC": SVC,  

        # "KNeighborsClassifier": KNeighborsClassifier,

        "GaussianNB": GaussianNB,

        "DecisionTreeClassifier": DecisionTreeClassifier,

    },

    "regression": {

        "LinearRegression": LinearRegression,

        "Ridge": Ridge,

        "Lasso": Lasso,

        "ElasticNet": ElasticNet,

        "RandomForestRegressor": RandomForestRegressor,

        "GradientBoostingRegressor": GradientBoostingRegressor,

        "AdaBoostRegressor": AdaBoostRegressor,

        "ExtraTreesRegressor": ExtraTreesRegressor,

        "XGBRegressor": XGBRegressor,

        "LGBMRegressor": LGBMRegressor,

        "CatBoostRegressor": CatBoostRegressor,

        # "SVR": SVR,

        "KNeighborsRegressor": KNeighborsRegressor,

        "DecisionTreeRegressor": DecisionTreeRegressor,

    }

}
 