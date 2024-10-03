# Use the official AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.9

# Install necessary Python libraries
RUN pip install --upgrade pip
RUN pip install numpy pyswarm scikit-learn joblib pandas tensorflow

# Copy your Lambda function code into the container
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model2.joblib ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (app.lambda_handler)
CMD ["app.lambda_handler"]
