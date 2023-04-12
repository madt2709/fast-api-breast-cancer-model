# Basic deployment of a model
Aim of this project is to train a logistic regression model on the Winsconcin breast cancer data set then deploy the results to a "production" ready API hosted on an EC2 istance. There are many more things we could add to this project such as:
- Security - HTTPS
- Running on startup
- Restarts
- Replication (the number of processes running)
- Memory
- Previous steps before starting

However, to keep this as basic as possible these will all be out of scope.

This will use tools such as Fast API, Docker and nginx to achieve the goal.

## How to set up locally

Clone the repo from github. 

Head to the repo on your machine.

Create virtual environment.

Download requirements:

``` bash 
pip install -r requirements.txt
```

Start server locally

``` bash
uvicorn main:app --reload
```

You will then find the docs here: http://127.0.0.1:8000/api/v1/docs.

## How to train model

Run

``` bash
python3 model/train.py
```

This will retrain the model and save the results to 'model\breast-cancer-v1.joblib'.

## How to deploy to EC2 instance

### Spin up an EC2 instance

There are 100's of tutorials on how to do this. I followed this [one](https://dev.to/genialkartik/aws-ec2-instance-57gb). 

I set up my instance with Ubuntu. The following commands will be aimed for Ubuntu users.

### Install docker and nginx on your server

SSH into your server and then download docker using the instructions found [here](https://docs.docker.com/engine/install/ubuntu/).

You can check it is properly installed by running:

``` bash
sudo docker run hello-world
```

Install nginx:

``` bash
sudo apt install nginx
```

### Clone the repo

Since you won't be writing to the repo, you can just use https:

``` bash
git clone https://github.com/madt2709/fast-api-breast-cancer-model.git
```

### Set up nginx

Configure your nginx instance. 

Create a config file: 

``` bash
sudo vi /etc/nginx/sites-enabled/fastapi-demo
```

Feel free to use another editor than vi. 

Paste the following into the file replacing the `<PUBLIC_IP>` with the public IP of your instance:

``` bash
server {
    listen 80;
    server_name <PUBLIC_IP>;
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

Save the file and restart nginx:

``` bash
sudo service nginx restart
```

### Start the docker container

Head to the repo.

Start the docker container in Detached mode. 

``` bash
sudo docker-compose up -d
```

It should now be up and running! Head to `<PUBLIC_IP>/api/v1/docs` to check it's working.

You can see an example at http://13.48.196.239/api/v1/docs (running at time of writing, this may not be the case in the future).




