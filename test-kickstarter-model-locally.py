from pandas import DataFrame
from qwak.model.tools import run_local
import importlib.util

# Add the path to the directory where the Python file is located
#sys.path.append('/Users/haha/Projects/qwak-onboarding/main')
#import crowdfunding_projects_classification_model


# Path to the Python file (replace with the actual path)
path_to_file = '/Users/haha/Projects/qwak-onboarding/main/crowdfunding-projects-classification-model.py'

# Load the spec for the module
spec = importlib.util.spec_from_file_location('crowdfunding_module', path_to_file)

# Create a module from the spec
crowdfunding_module = importlib.util.module_from_spec(spec)

# Execute the module to make its contents available
spec.loader.exec_module(crowdfunding_module)


# Now you can use functions, classes, etc. from the module
#module.some_function()


if __name__ == '__main__':
    # Create a new instance of the model
    m = crowdfunding_module.CrowdfundingProjectsClassificationModel()

    # Define the columns
    columns = [
            "project_id", "launched", "current", "start_timestamp", "end_timestamp", "date",
            "category", "currency",
            "state", "usd_pledged",
            "country", "name",
            "goal", "main_category",
            "backers", "pledged",
            "deadline"
        ]

    # Define the data
    data = [
        [
            "776758468", "2010-04-20 05:58:42.000000 UTC", "true", "2010-04-20 05:58:42.000000 UTC", None,
            "2010-04-20", "Theater", "USD", "successful", "3000", "US",
            "Silent Bugler Productions Presents Michael McClure's The Beard", "3000", "Theater", "8", "3000",
            "2010-06-09 05:59:00.000000 UTC"
        ]
    ]

    # Create the DataFrame
    df = DataFrame(data, columns=columns).to_json()
    print("Predicting for \n")
    print(df)

    # Run local inference using the model
    prediction = run_local(m, df)
    print(prediction)     