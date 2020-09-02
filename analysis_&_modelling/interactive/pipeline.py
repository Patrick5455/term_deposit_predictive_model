#%%writefile ../scripts/project_package/model_package/pipeline.py 
from sklearn.pipeline import FeatureUnion, Pipeline  
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline, make_pipeline
