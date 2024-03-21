echo "Starting up the sytem."
cd ./src/app_corporate_culture_measurer/
python -m flask --app app_corporate_culture_measurer --debug run --host=0.0.0.0 --port=8080 