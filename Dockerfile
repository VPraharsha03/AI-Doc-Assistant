# start from python base image
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# change working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN uv pip install --no-cache-dir --upgrade -r /code/requirements.txt --system

# add python code
COPY ./app/ /code/app/

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]