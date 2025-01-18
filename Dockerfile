FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS build

# Define arguments
ARG CORES

# Install necessary packages
RUN apt update && apt install -y \
  make
RUN apt upgrade -y && apt clean && rm -rf /var/lib/apt/lists/*

# Instal python packages
RUN mamba install -y \
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
  scikit-learn \
  seaborn \
  torchvision \
  opencv \
  dvc

RUN mamba update --all -y && mamba clean --all -y

# Grant read, write, and execute permissions to all users
RUN find /opt/conda -type f -print0 | xargs -0 -P ${CORES} chmod 777 \
  && find /opt/conda -type d | xargs -P ${CORES} chmod 777

FROM build AS dev

# Remove default user
RUN if [ -n "$(id -u ubuntu)" ]; then userdel -r ubuntu; fi

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

RUN apt upgrade -y && apt clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
  && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} -s /bin/zsh
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Change to non-root user
USER ${USER_NAME}

# Setup zsh shell with Oh-My-Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ENTRYPOINT [ "zsh" ]
