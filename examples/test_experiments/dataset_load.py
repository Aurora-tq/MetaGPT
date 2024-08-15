import os
def dataload(dataset_name):
    # Define data paths and user requirements for different datasets
    data_info = {
        "santander_customers":{
            "data_path" : '/Users/aurora/Desktop/ml_benchmark/06_santander-customer-transaction-prediction',
            "user_requirement" :"This is a customers financial dataset. Your goal is to predict which customers will make a specific transaction in the future. The target column is target. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report AUC on the eval data. Don't need to plot."
        },
        "Titanic":{
            "data_path" : "/Users/aurora/Desktop/ml_benchmark/04_titanic" ,
            "user_requirement":"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target.Please recommend at least five models and evaluate their performance separately, and finally input the optimal result. To predict the target, the  Report accuracy on the eval data. Don't need to plot. "
        },
        'ICR':{
            "data_path":"/Users/aurora/Desktop/ml_benchmark/07_icr-identify-age-related-conditions" ,
            "user_requirement":" ICR dataset is a medical dataset with over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions. Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. The target column is Class. Report F1 Score on the eval data. Don't need to plot."
        },
        'House Price':{
            "data_path":"/Users/aurora/Desktop/ml_benchmark/05_house-prices-advanced-regression-techniques" ,
            "user_requirement":"This is a house price dataset, your goal is to predict the sale price of a property based on its features. Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. Report RMSE between the logarithm of the predicted value and the logarithm of the observed sale prices on the eval data. The target column is 'SalePrice'. Please do not include any processing of the target column in the data preprocessing and feature engineering stages. Don't need to plot."
        }

    }
    if dataset_name in data_info:
        return data_info[dataset_name]['data_path'], data_info[dataset_name]['user_requirement']
    else:
        raise ValueError(f"Dataset {dataset_name} not found in data info")


def dataload(dataset_name, config):

    datasets_dir = config['datasets_dir']
    if dataset_name in config['datasets']:
        dataset = config['datasets'][dataset_name]
        data_path = os.path.join(datasets_dir, dataset['dataset'])
        user_requirement = dataset['user_requirement']
        return data_path, user_requirement
    else:
        raise ValueError(f"Dataset {dataset_name} not found in config file. Available datasets: {config['datasets'].keys()}")

