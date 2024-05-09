

This repository is the solution to the hiring challenge for mlops intern at Spacesense.

1. It expected the image segmentation code should be served through a FastAPI server. 
2. Dockerize the whole FastAPI server.

The solution can executed by two ways

1. Locally
2. Docker

## Locally
For running it locally create an anaconda environment

1. `conda create -n spacesense python=3.9`
2. `conda activate spacesense`
3. `pip install -r requirements.txt`
4. `uvicorn server:app --host 0.0.0.0 --port 8000`

5. In another terminal run 

```
curl -L -F "file=@resources/dog.jpg" http://127.0.0.1:8000/segmentation -o "test.png"
```

You will get the ouput image as test.png.

## Docker
For Docker 
1. Install docker first in your machine.
2. Go to docker directory using `cd docker`
3. Now build the docker image using `docker build -t imageseg  .`
4. Run the container `docker container run -d   -p 8000:8000 imageseg`
5. Test the server by `cd ..` and then  `curl -L -F "file=@resources/dog.jpg" http://127.0.0.1:8000/segmentation -o "test.png"`

6. You can even directly run the already built image
`docker container run -d   -p 8000:8000 pronoob007/spacesense-image-seg:latest`
Obviously it should be tested the same way you built the container.
