import os
from sagemaker.sklearn import SKLearnModel
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig

# Set these environment variables or replace with your values
S3_BUCKET = "bucket-857196"        # your bucket name
MODEL_S3_PATH = f"s3://{S3_BUCKET}/sagemaker/mobile_price_classification/sklearncontainer/model/model.tar.gz"
REGION = "us-west-2"
ROLE = "arn:aws:iam::691879165105:role/service-role/AmazonSageMaker-ExecutionRole-20250517T013314"

sess = sagemaker.Session()

sklearn_model = SKLearnModel(
    model_data=MODEL_S3_PATH,
    role=ROLE,
    entry_point="script.py",
    framework_version="0.23-1",
    sagemaker_session=sess
)

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://bucket-857196/sagemaker/mobile_price_classification/sklearncontainer/datacapture',
    capture_options=["Input", "Output"]
)

predictor = sklearn_model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    data_capture_config=data_capture_config,
)

print(f"Model deployed at endpoint: {predictor.endpoint_name}")
