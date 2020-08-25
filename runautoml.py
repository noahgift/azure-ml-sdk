from azureml.core import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig

# Create Workspace
ws = Workspace.get(name='Azure-ML-Workspace',
                      subscription_id='a59af59a-605a-4b0e-968f-a30ad6bb7ad5',
                      resource_group='cloud-shell-storage-eastus'
                     )

# Create AutoMLConfig
automl_config = AutoMLConfig(task="classification",
                             X=your_training_features,
                             y=your_training_labels,
                             iterations=30,
                             iteration_timeout_minutes=5,
                             primary_metric="AUC_weighted",
                             n_cross_validations=5
                            )

# Create an Experiment
experiment = Experiment(ws, "automl_test_experiment")
run = experiment.submit(config=automl_config, show_output=True)

# Find the best model
best_model = run.get_output()
y_predict = best_model.predict(X_test)