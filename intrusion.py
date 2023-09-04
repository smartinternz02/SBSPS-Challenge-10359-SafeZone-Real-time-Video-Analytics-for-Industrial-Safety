from sdcclient import IbmAuthHelper, SdMonitorClient
import cv2
from cvzone.PoseModule import PoseDetector
detector=PoseDetector()
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
# IBM Cloud Monitoring configuration
URL = 'https://jp-osa.monitoring.cloud.ibm.com'  # Replace with your IBM Cloud Monitoring URL
APIKEY = 'H4jpw9BfaZ6Ry7GNfW70QbgRWdd7tShMLjEPwFdMfG8y'  # Replace with your IAM API key
GUID = 'e6b98a05-f37f-454b-9d02-03e1d9cac68b'  # Replace with your Monitoring instance GUID

# Authenticate using IBM Cloud Monitoring Python client
ibm_headers = IbmAuthHelper.get_headers(URL, APIKEY, GUID)
sdclient = SdMonitorClient(sdc_url=URL, custom_headers=ibm_headers)

# Add the notification channels to send an alert when the alert is triggered
notify_channels = [
    {
        'type': 'EMAIL',
        'emailRecipients': ['sakshishosamani@gmail.com']  # Add recipient email addresses
    }
]

# Get the IDs of the notification channels that you have configured
res = sdclient.get_notification_ids(notify_channels)
if not res[0]:
    print("Failed to fetch notification channel IDs")
else:
    print(res[0])

notification_channel_ids = res[1]  # Extract notification channel IDs from the response
print(notification_channel_ids)
while True:
    success,img=cap.read()
    img=detector.findPose(img)
    imlist,bbox=detector.findPosition(img)
    l=len(imlist)
    cv2.imshow("Output",img)
    q=cv2.waitKey(1)
    if q==ord('q'):
        break
# Create and define the alert details
alert_params = {
    'name': 'Test11Alert',
    'description': 'This is a test alert',
    'severity': 3,
    'for_atleast_s': 60,
    'condition':'len(imlist)>0',
    'notify':notification_channel_ids,
    'enabled': True
}
# Create the alert
res = sdclient.create_alert(**alert_params)
print(res)
if res[0]:
    print("Alert created successfully")
else:
    print("Alert creation failed")
