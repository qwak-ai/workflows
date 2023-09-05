from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, InferenceOutput, ExplicitFeature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from datetime import datetime
import pandas as pd
import qwak
import os
from main.utils import get_data_from_feature_store
from main.utils import syllable_count

'''
    We're building the model from this Kaggle experiment
    https://www.kaggle.com/code/shivamb/an-insightful-story-of-crowdfunding-projects/notebook
'''

class CrowdfundingProjectsClassificationModel(QwakModel):

    def __init__(self):
        self.params = {
            'n_estimators': int(os.getenv('n_estimators', 50)),
            'random_state': int(os.getenv('random_state', 0))
        }

        self.model = RandomForestClassifier(**self.params)
        self.encoder = LabelEncoder()
        self.features = []

        qwak.log_param(self.params)

    def build(self):

        feature_set = "kickstarter-projects-features"   
        query=f"""
                SELECT  project_id,
                        launched,
                        "kickstarter-projects-features.category" as category,
                        "kickstarter-projects-features.currency" as currency,
                        "kickstarter-projects-features.state" as state,
                        "kickstarter-projects-features.usd_pledged" as usd_pledged,
                        "kickstarter-projects-features.country" as country,
                        "kickstarter-projects-features.name" as name,
                        "kickstarter-projects-features.goal" as goal,
                        "kickstarter-projects-features.main_category" as main_category,
                        "kickstarter-projects-features.pledged" as pledged,
                        "kickstarter-projects-features.backers" as backers,
                        "kickstarter-projects-features.deadline" as deadline
                FROM "qwak_feature_store_00008a4a_ee42_4afb_bf3f_01c48a3429ce"."offline_feature_store_{feature_set.replace("-", "_")}"
                WHERE launched > date('2012-01-01') 
                """


        project_features = get_data_from_feature_store(query)

        print ("Total Projects: ", project_features.shape[0], "\nTotal Features: ", project_features.shape[1])
        print(project_features.sample(100, replace=True))

        project_features = self.feature_engineering_and_processing(project_features)
        
        ## define predictors and label 
        ## Getting the original columns order
        label = project_features.state
        self.features = [c for c in project_features.columns if c not in ["state", "name"]]

        ## prepare training and testing dataset
        print("Splitting dataset")
        X_train, X_test, y_train, y_test = train_test_split(project_features[self.features], label, test_size = 0.025, random_state = 2)

        ## train a random forest classifier 
        print("Training the model")
        self.model = self.model.fit(X_train, y_train)

        print("Testing the predictions")
        y_pred = self.model.predict(X_test)

        # Log metrics into Qwak
        accuracy = accuracy_score(y_test, y_pred)
        qwak.log_metric({"val_accuracy": accuracy})
        

    def schema(self) -> ModelSchema:
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="Project_Id", type=str),
                ExplicitFeature(name="Name", type=str),
                ExplicitFeature(name="Category", type=str),
                ExplicitFeature(name="Main_category", type=str),
                ExplicitFeature(name="Currency", type=str),
                ExplicitFeature(name="Deadline", type=datetime),
                ExplicitFeature(name="Goal", type=int),
                ExplicitFeature(name="Launched", type=datetime),
                ExplicitFeature(name="Pledged", type=int),
                ExplicitFeature(name="State", type=str),
                ExplicitFeature(name="Backers", type=int),
                ExplicitFeature(name="Country", type=str),
                ExplicitFeature(name="USD_Pledged", type=int)
            ],
            outputs=[
                InferenceOutput(name="Successful", type=int)
            ])
        return model_schema
    
    @qwak.api()
    def predict(self, df: DataFrame) -> DataFrame:
        """
            The predict(df) method is the actual inference method.
        """
        prediction_data = self.feature_engineering_and_processing(df)

        if (prediction_data.empty):
            print("Prediction was canceled for this project that I didn't know how to print its ID\n")
            return pd.DataFrame(
                df,
                columns=['state']
            ).rename(columns={'state': 'successful'})

        # Reformatting the prediction data order 
        prediction_data = df.drop(
            ['state', 'name'], axis=1
        ).reindex(columns=self.features)

        predictions = self.model.predict(prediction_data)
    
        return pd.DataFrame(
            predictions,
            columns=['state']
        ).rename(columns={'state': 'successful'})
    

    
    # this method will be called both in training and predict to manipulate the raw data into features
    def feature_engineering_and_processing(self, raw_input) -> DataFrame:

        #Handle missing values, filter projects with USD currency and the ones which finished FAILED or SUCCESSFUL
        project_features_df = raw_input.dropna()
        project_features_df = project_features_df[project_features_df["currency"] == "USD"]
        project_features_df = project_features_df[project_features_df["state"].isin(["failed", "successful"])]

        #I will need to update the columns with the name from the feature store
        project_features_df = project_features_df.drop(["backers", "project_id", "currency", "country", "pledged", "usd_pledged"], axis = 1)
        
        ## feature engineering
        project_features_df["syllable_count"]   = project_features_df["name"].apply(lambda x: syllable_count(x))
        project_features_df['launched']         = pd.to_datetime(project_features_df['launched'])
        project_features_df["launched_month"]   = project_features_df["launched"].dt.month
        project_features_df["launched_week"]    = project_features_df["launched"].dt.isocalendar().week
        project_features_df["launched_day"]     = project_features_df["launched"].dt.weekday
        project_features_df["is_weekend"]       = project_features_df["launched_day"].apply(lambda x: 1 if x > 4 else 0)
        project_features_df["num_words"]        = project_features_df["name"].apply(lambda x: len(x.split()))
        project_features_df["num_chars"]        = project_features_df["name"].apply(lambda x: len(x.replace(" ","")))
        project_features_df['deadline']         = pd.to_datetime(project_features_df['deadline'])
        project_features_df["duration"]         = project_features_df["deadline"] - project_features_df["launched"]
        project_features_df["duration"]         = project_features_df["duration"].apply(lambda x: int(str(x).split()[0]))
        project_features_df["state"]            = project_features_df["state"].apply(lambda x: 1 if x=="successful" else 0)

        ## label encoding the categorical features
        project_features_df = pd.concat([project_features_df, pd.get_dummies(project_features_df["main_category"])], axis = 1)
        
        for c in ["category", "main_category"]:
            project_features_df[c] = self.encoder.fit_transform(project_features_df[c])

        ## Generate Count Features related to Category and Main Category
        t2 = project_features_df.groupby("main_category").agg({"goal" : "mean", "category" : "sum"})
        t1 = project_features_df.groupby("category").agg({"goal" : "mean", "main_category" : "sum"})
        t2 = t2.reset_index().rename(columns={"goal" : "mean_main_category_goal", "category" : "main_category_count"})
        t1 = t1.reset_index().rename(columns={"goal" : "mean_category_goal", "main_category" : "category_count"})
        project_features_df = project_features_df.merge(t1, on = "category")
        project_features_df = project_features_df.merge(t2, on = "main_category")

        project_features_df["diff_mean_category_goal"] = project_features_df["mean_category_goal"] - project_features_df["goal"]
        project_features_df["diff_mean_category_goal"] = project_features_df["mean_main_category_goal"] - project_features_df["goal"]

        project_features_df = project_features_df.drop(["launched", "deadline"], axis = 1)
        project_features_df[[c for c in project_features_df.columns if c != "name"]].head()

        return project_features_df
