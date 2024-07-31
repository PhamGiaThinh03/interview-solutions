## Run Dockerfile:
- docker build -t thinhpg.
- docker run -it --rm -p 8010:8000 --name=thinhpg -v ./:/app fastapi-thinhpg-api

## Run an ASGI
 - uvicorn main:api.app --host 0.0.0.0 --port 8000 

![Dockerfile](../assets/images/uvicorn.png)

## Run pytest
- pytest

![Pytest](../assets/images/pytest.png)
![Postman](../assets/images/postman.png)


## CI/CD
 - POST: http://localhost:8010/predict

![CICD](../assets/images/cicd.png)
![Dockerhub](../assets/images/dockerhub.png)
