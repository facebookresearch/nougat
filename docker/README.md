## Prerequisites
Ensure you have Docker installed on your machine. 
And you must also have NVIDIA CUDA and CuDNN installed in your machine. 

Then, you must check your machine's CUDA version.
```sh
nvcc -V
```

You must change base image name and pytorch version compatible with your **CUDA version**. 

## Building the Docker Image
Clone this repository and navigate into the current directory(nougat/docker). You can build the Docker image by running:
```sh
docker build -t <image-name> .
```
Replace <image-name> with a name of your choice. This will be used to refer to the image later.
Please be patient as this operation can take a while. It needs to pull the CUDA-capable image from NVIDIA’s Docker repository and install several libraries.
Image size will be about 17GB.


## Running the Docker Container
You can run your Docker container with the following command:
```sh
docker run -it -d -p <your-port>:8503 --gpus all <image-name>
```
Replace <your-port> with the port number you wish to expose on your host machine to access the nougat API server.
This can be any valid port number. Replace <image-name> with the name you chose earlier during the build step.


## Testing the API Server
Once the Docker container is running, you can access the nougat API server.
You can easily check connection by running:
```sh
curl -X 'GET' \
  'http://127.0.0.1:<your-port>/'
```
It can be take a while for loading API server, because the server have to download nougat model at startup.

If connection is successful, you can get response looks like this.
```
{"status-code":200,"data":{}}
```

## Using the API Server
To get a prediction of a PDF file by making a POST request to `http://127.0.0.1:<your-port>/predict/`. It also accepts parameters `start` and `stop` to limit the computation to select page numbers (boundaries are included).

The response is a string with the markdown text of the document.

```sh
curl -X 'POST' \
  'http://127.0.0.1:<your-port>/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<PDFFILE.pdf>;type=application/pdf'
```
To use the limit the conversion to pages 1 to 5, use the start/stop parameters in the request URL: 
`http://127.0.0.1:<your-port>/predict/?start=1&stop=5`


