---
title: "tutorial.md"
author: "Marco Wang"
date: "2026-03-18"
Disclaimer: This documentation is generated with the help of Gemini3
---
# How to Launch the Linguistica Application via Docker
You can easily launch the entire Linguistica application (both the backend API and the frontend) entirely inside a Docker container. 

## Step 1: Load the Docker Image
First, load the image into your local Docker instance. Download the `linguistica.tar` file from the [Google Drive link](https://drive.google.com/file/d/1diEQX4MGrMNv63x_7jshWO3M6gjoFNa1/view?usp=sharing) and save it to your computer. Then, open your terminal in the folder containing `linguistica.tar` and run:

```bash
docker load -i linguistica.tar
```

*What this does:* This command tells Docker to read the `.tar` file and extract the image layers and tags. Once it finishes, the image named `linguistica` will be available in your local Docker image registry.

## Step 2: Run the Docker Container
Once the image is loaded, spin up the application container:

```bash
docker run -p 8000:8000 linguistica
```

*What this does:* 
- `docker run` tells Docker to start a new container from the `linguistica` image.
- `-p 8000:8000` tells Docker to map port 8000 on your local to port 8000 inside the Docker container. 

## Step 3: Access the Application
You can now open your web browser and navigate directly to:
**http://127.0.0.1:8000** 
