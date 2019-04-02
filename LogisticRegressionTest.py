from __future__ import print_function

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow.sklearn

import csv


if __name__ == "__main__":
	#set the server tracking uri
	mlflow.set_tracking_uri("http://127.0.0.1:5000")

	#set the parameters
	mlflow.log_param("regularization", "l2")

	X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
	y = np.array([0, 0, 1, 1, 1, 0])
	lr = LogisticRegression()
	lr.fit(X, y)

	score = lr.score(X, y)
	print("Score: %s" % score)

	lr_weights = str(lr.coef_)
	print(lr.coef_)

	# log the metric(scores, key-value)
	mlflow.log_metric("score", score)

	csv_path = '/tmp/logistic_regression.csv'
	csvData = [['col1','col2','col3'],[2,3,4]]
	with open(csv_path, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(csvData)

	# example save artifact
	mlflow.log_artifact(csv_path)

	#save the model
	mlflow.sklearn.log_model(lr, "model")
	print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

	mlflow.end_run()

