
# Sort images by visual similarities

Sort a folder of images by grouping together the images that are visually similar

## Run Locally

Clone the project

```bash
  git clone https://github.com/tachillon/sort-images-by-similarities-and-look-alike.git
```

Go to the project directory

```bash
  cd sort-images-by-similarities-and-look-alike
```

Install dependencies

```bash
  Install Docker: https://docs.docker.com/get-docker/
```

Build the docker container

```bash
  docker build -t <container_name>:<container_tag> .
```
Caution: might break if Tensorflow updates its base Tensorflow docker image

Run the program

```bash
  docker run --rm -it -u $(id -u) -w /tmp -v ${PWD}:/tmp <container_name>:<container_tag> python3 image_similarity.py --input_dir ./imgs --use_tfhub_model true
```
