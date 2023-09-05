from datetime import datetime
import qwak.feature_store.feature_sets.read_policies as ReadPolicy
from qwak.feature_store.feature_sets import batch
from qwak.feature_store.feature_sets.transformations import SparkSqlTransformation
from qwak.feature_store.entities.entity import Entity

project = Entity(
    name='project_id',
    description='A Registered project in the Kickstarter platform',
)

@batch.feature_set(
    name="kickstarter-projects-features",
    entity="project_id",
    data_sources = {
        "Kickstarter_Dataset": ReadPolicy.NewOnly
    }
)
@batch.metadata(
    owner="Grig Duta",
    display_name="Kickstarter Project Success Features",
    description="Features describing Kickstarter projects ",
)
@batch.scheduling(cron_expression="0 0 * * *")
@batch.backfill(start_date=datetime(2000, 1, 1)) 

def project_features():
    return SparkSqlTransformation(
        """
        SELECT  ID as project_id,
                NAME,
                CATEGORY,
                MAIN_CATEGORY,
                CURRENCY,
                DEADLINE,
                GOAL,
                LAUNCHED,
                PLEDGED,
                STATE,
                BACKERS,
                COUNTRY,
                USD_PLEDGED
        FROM Kickstarter_Dataset
        """
    )
