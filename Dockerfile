FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS build

# Remove default user
RUN if [ -n "$(id -u ubuntu)" ]; then userdel -r ubuntu; fi

# Install necessary packages
RUN apt update && apt install -y \
  build-essential

# # Instal python packages
RUN conda install -y \
  -c conda-forge \
  -c pytorch \
  -c nvidia \
  pytest \
  pylint \
  black \
  isort \
  notebook \
  pandas \
  matplotlib \
  scikit-learn

RUN conda update --all -y

FROM build AS dev

# Define arguments
ARG USER_NAME
ARG USER_UID
ARG USER_GID

# Install necessary packages            
RUN apt update && apt install -y \
  sudo \
  zsh \
  fzf \
  xclip \
  xsel \
  curl \
  wget \
  git \
  tree \
  tmux \
  vim

RUN apt upgrade -y
RUN apt clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
  && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} -s /bin/zsh
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Change owner of conda to non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} /opt/conda

# Change to non-root user
USER ${USER_NAME}

# Setup zsh shell with Oh-My-Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ENTRYPOINT [ "zsh" ]
