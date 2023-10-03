import flask
from flask import request
import os
from bot import ObjectDetectionBot
import boto3
from botocore.exceptions import ClientError

app = flask.Flask(__name__)


# TODO load TELEGRAM_TOKEN value from Secret Manager
def get_secret():

    secret_name = "Moshiko_Token"
    region_name = "eu-north-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return secret


TELEGRAM_TOKEN = get_secret()
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


@app.route(f'/results/', methods=['GET'])
def results():
    prediction_id = request.args.get('predictionId')

    # TODO use the prediction_id to retrieve results from DynamoDB and send to the end-user
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    table = dynamodb.Table('Moshiko_Yolo')

    try:
        response = table.get_item(
            Key={
                'prediction_id': prediction_id
            }
        )
        item = response.get('Item')
        if item:
            text_results = item.get('text_results')
            bot.send_text(prediction_id, text_results)
            return 'Results sent to the user'
        else:
            return 'No results found for the given prediction ID'
    except Exception as e:
        return f'Error: {str(e)}'


@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)

    app.run(host='0.0.0.0', port=8443)
