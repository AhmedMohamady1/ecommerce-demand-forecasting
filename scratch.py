import pandas as pd
import numpy as np
import os
import pickle
import boto3
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

load_dotenv(".env")
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
s3 = boto3.client("s3", endpoint_url=minio_endpoint, aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"), aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"))

obj = s3.get_object(Bucket="ecommerce-lake", Key="gold/models/prophet/prophet_models.pkl")
models = pickle.loads(obj['Body'].read())

df = pd.read_csv("data/2017_test_data.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.isocalendar().week

test_weeks = df[['store', 'item', 'year', 'week']].drop_duplicates()
predictions = []

def _week_to_date(year, week):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    return jan1 + pd.Timedelta(days=(week - 1) * 7)

for (s, i), grp in test_weeks.groupby(['store', 'item']):
    model = models.get((s, i))
    if model is not None:
        test_prophet = pd.DataFrame({"ds": [_week_to_date(r["year"], r["week"]) for _, r in grp.iterrows()]})
        forecast = model.predict(test_prophet)
        grp = grp.copy()
        grp['sales_prediction'] = forecast['yhat'].values
        predictions.append(grp)

preds_df = pd.concat(predictions, ignore_index=True)

import sys
from config.spark_config import get_spark
spark = get_spark("Test")
gold_spark = spark.read.parquet("s3a://ecommerce-lake/gold/features/")
gold_df = gold_spark.select("year", "week_of_year", "store", "item", "weekly_sales").toPandas()
gold_df = gold_df.rename(columns={"week_of_year": "week"})

merged = pd.merge(preds_df, gold_df, on=["year", "week", "store", "item"], how="inner")
print(f"Merged rows: {len(merged)}")

rmse = np.sqrt(mean_squared_error(merged["weekly_sales"], merged["sales_prediction"]))
print(f"App RMSE: {rmse}")

# Now let's calculate Prophet RMSE exactly like train_evaluate.py does
test_spark_df = gold_spark.filter(gold_spark.year >= 2017).toPandas()
test_spark_df = test_spark_df.rename(columns={"week_of_year": "week"})
test_spark_df = test_spark_df.sort_values(by=["store", "item", "year", "week"]).reset_index(drop=True)

te_predictions = []
for (s, i), grp in test_spark_df.groupby(['store', 'item']):
    model = models.get((s, i))
    if model is not None:
        test_prophet = pd.DataFrame({"ds": [_week_to_date(r["year"], r["week"]) for _, r in grp.iterrows()]})
        forecast = model.predict(test_prophet)
        grp = grp.copy()
        grp['sales_prediction'] = forecast['yhat'].values
        te_predictions.append(grp)
        
te_preds_df = pd.concat(te_predictions, ignore_index=True)
te_merged = pd.merge(te_preds_df, gold_df, on=["year", "week", "store", "item"], how="inner")
te_rmse = np.sqrt(mean_squared_error(te_merged["weekly_sales"], te_merged["sales_prediction"]))
print(f"Train_evaluate RMSE: {te_rmse}")

diff = pd.merge(merged, te_merged, on=["store", "item", "year", "week"])
diff['diff'] = diff['sales_prediction_x'] - diff['sales_prediction_y']
print(f"Max absolute difference in predictions: {diff['diff'].abs().max()}")
print(f"Rows with diff > 1e-5: {len(diff[diff['diff'].abs() > 1e-5])}")
if len(diff[diff['diff'].abs() > 1e-5]) > 0:
    print(diff[diff['diff'].abs() > 1e-5].head())
