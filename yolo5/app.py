import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import json
from flask import request

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']
s3 = boto3.client('s3')
s3_client = boto3.client('s3', region_name='eu-north-1')
sqs_client = boto3.client('sqs', region_name='eu-north-1')
dynamodb_client = boto3.client('dynamodb', region_name='eu-north-1')


def consume():
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            message_data = json.loads(message)

            # Receives a URL parameter representing the image to download from S3
            img_name = message_data.get('imgName')  # TODO extract from `message`
            chat_id = message_data.get('chatId')  # TODO extract from `message`
            filename = img_name.split('/')[-1]
            local_dir = 'photos/'  # str of dir to save to
            os.makedirs(local_dir, exist_ok=True)  # make sure the dir exists
            original_img_path = filename
            s3.download_file(images_bucket, img_name, original_img_path)  # TODO download img_name from S3, store the
            # local image path in original_img_path

            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels The predicted image typically includes bounding
            # boxes drawn around the detected objects, along with class labels and possibly confidence scores.
            predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

            # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
            predicted_img_name = f'predicted_{filename}'  # assign the new name
            os.rename(f'/usr/src/app/static/data/{prediction_id}/{filename}',
                      f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}')  # rename the file before upload
            s3_path_to_upload_to = '/'.join(
                img_name.split('/')[:-1]) + f'/{predicted_img_name}'  # assign the path on s3 as str
            file_to_upload = f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}'  # assign the path locally as str
            s3.upload_file(file_to_upload, images_bucket,
                           s3_path_to_upload_to)  # upload the file to same path with new name s3
            os.rename(f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}',
                      f'/usr/src/app/static/data/{prediction_id}/{filename}')  # rename the file back after upload

            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

                prediction_summary = {
                    'prediction_id': prediction_id,
                    'original_img_path': original_img_path,
                    'predicted_img_path': predicted_img_path,
                    'labels': labels,
                    'time': time.time()
                }

                # TODO store the prediction_summary in a DynamoDB table
                # TODO perform a GET request to Polybot to `/results` endpoint

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
