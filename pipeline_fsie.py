#!/usr/bin/env python
# coding: utf-8

# In[152]:


from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
import sagemaker
import boto3
import os

sagemaker_role = "arn:aws:iam::691879165105:role/service-role/AmazonSageMaker-ExecutionRole-20250528T020139"


# In[153]:


# --- Set up session and role ---
sess = sagemaker.session.Session()
pipeline_session = PipelineSession()
role = sagemaker_role



# In[154]:


# --- Parameters ---
input_data_uri = ParameterString(name="InputData", default_value="s3://bucket-857196/sagemaker/mobile_price_classification/sklearncontainer/mob_price_classification.csv")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
n_estimators = ParameterInteger(name="NEstimators", default_value=100)



# In[155]:


# --- Processing Step: Split CSV into train/test/baseline ---
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="preprocessing-job",
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="PreprocessingStep",
    processor=sklearn_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data_uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        sagemaker.processing.ProcessingOutput(output_name="baseline", source="/opt/ml/processing/baseline")
    ],
    code="preprocessing.py"
)



# In[156]:


# --- Training Step ---

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    # created above
    entry_point="script.py",

    # ARN of a new sagemaker role (ARN of new user does not work)
    role=role,

    # creates instance inside the Sagemaker machine
    instance_count=1,
    instance_type="ml.m5.large",

    # framework version present in the documentation, declared above
    framework_version=FRAMEWORK_VERSION,

    # name of folder after model has been trained
    base_job_name="training-job",

    # hyperparameters to the RF classifier
     hyperparameters={
        "n_estimators": 100,
        "random_state": 0,
    },
    use_spot_instances = True,
    max_wait = 7200,
    max_run = 3600
)

training_step = TrainingStep(
    name="TrainingStep",
    estimator=sklearn_estimator,
    inputs={
        "train": processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        "test": processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
    }
)



# In[157]:


# --- Model Evaluation Step (Accuracy) ---
# This uses a simple evaluation script to calculate accuracy on test set

eval_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="evaluation-job",
    sagemaker_session=pipeline_session
)

from sagemaker.workflow.properties import PropertyFile

evaluation_report = PropertyFile(
    name="EvaluationReport",  # This is the evaluation report name
    output_name="evaluation",  # Must match ProcessingOutput name
    path="evaluation.json"     # Path inside the output directory
)


evaluation_step = ProcessingStep(
    name="EvaluationStep",
    processor=eval_processor,
    code="evaluation.py",
    property_files=[evaluation_report],
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        sagemaker.processing.ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ]
)



# In[158]:


from sagemaker.workflow.functions import Join
from sagemaker.model_metrics import MetricsSource, ModelMetrics

# --- Model Registration Step ---
evaluation_output = evaluation_step.outputs[0]  # assuming first output is 'evaluation'


model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="/", values=[
            evaluation_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
            "evaluation.json"
        ]),
        content_type="application/json"
    )
)

register_step = RegisterModel(
    name="RegisterModel",
    estimator=sklearn_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="Model-857196",
    approval_status=model_approval_status,
   model_metrics=model_metrics
)


# In[159]:


from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Define the threshold
accuracy_threshold = 0.8

# Create a condition
cond_accuracy = ConditionGreaterThan(
    left=JsonGet(
        step_name=evaluation_step.name,  
        property_file=evaluation_report,  # PropertyFile object from evaluation step
        json_path="binary_classification_metrics.accuracy.value",  # Path to accuracy in the JSON
    ),
    right=accuracy_threshold
)

# Define the condition step
condition_step = ConditionStep(
    name="CheckAccuracyBeforeRegister",
    conditions=[cond_accuracy],
    if_steps=[register_step]   # Run registration step if condition is met
)


# In[160]:


# --- Pipeline Definition ---
pipeline = Pipeline(
    name="pipeline-857196-1",
    parameters=[input_data_uri, model_approval_status, n_estimators],
    #steps=[processing_step, training_step, evaluation_step, register_step],
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session
)


# In[161]:


# --- Upsert and Start Execution (optional) ---
pipeline.upsert(role_arn=role)


# In[162]:


execution = pipeline.start()


# In[ ]:




