FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME \
    && mkdir -p /home/$USERNAME/app \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

USER $USERNAME

ENV PATH="/home/appuser/.local/bin:$PATH"
WORKDIR /home/$USERNAME/app

COPY --chown=$USERNAME:$USERNAME . /home/$USERNAME/app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
