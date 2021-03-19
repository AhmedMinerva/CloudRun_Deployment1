To run locally:
- Make env with Python<=3.8 (for TF1.15 to work)
- Install the follwing packages (from Dockerfile): Flask gunicorn tensorflow==1.15 opencv-python Pillow requests
- For opencv to work (on Linux), you need to install the following packages libglib2.0-0 libsm6 libxext6 libxrender-dev. This can be done with sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev
- Run app.py and you can send post requests to port 8080.

To deploy to Google Cloud Run:
Assuming gcloud is properly set up,
1. First, make container,
```model/saved_models/vgg19_no_fc.npy
.git
gcloud builds submit --tag gcr.io/PROJECT-ID/MODEL-NAME
```
--> Replace PROJECT-ID with the id of current project in cloud console. MODEL-NAME is the name of project to store on cloud
2. Deploy :)
```
gcloud builds submit --tag gcr.io/PROJECT-ID/MODEL-NAME
```
Follow simple steps for configuration, but should be fine after that (select location, security, etc.)
