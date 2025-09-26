# Use NVIDIA CUDA 11.7 base image with Ubuntu 22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set default shell
SHELL ["/bin/bash", "-c"]

ARG USERNAME=zuleikarg
ARG USER_UID=1000
ARG USER_GID=1000

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools and ROS dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    python3-setuptools \
    python3-pip \
    python3-empy \
    python3-nose \
    python3-pytest \
    python3-pkg-resources \
    python3-opencv \
    python3-dev \
    python3-flake8-docstrings \
    python3-pytest-cov \
    ninja-build \
    locales \
    nano \
    fim \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 1.13 for CUDA 11.7
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==3.4.18.65 \
    colcon-common-extensions \
    cython \
    pillow \
    pycocotools \
    matplotlib \
    wheel==0.45.1

RUN pip3 uninstall spatial-correlation-sampler==0.4.0

# Add ROS 2 apt repository
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && apt-get update

# Forcefully remove conflicting packages before installing ROS 2 Desktop
RUN apt-get purge -y python3-catkin-pkg-modules python3-rospkg-modules python3-rosdistro-modules || true

# Install ROS 2 Desktop (RViz included)
RUN apt-get install -y --no-install-recommends ros-humble-desktop

RUN apt-get install -y --no-install-recommends python3-rosdep2

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Source ROS 2 setup script in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Setup environment variables for ROS 2 and CUDA
ENV ROS_DISTRO=humble
ENV PATH=/opt/ros/$ROS_DISTRO/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda


RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --no-install-recommends \
    cmake \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    iproute2 vim htop \
    net-tools \
    ca-certificates \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-nav-msgs \
    bash-completion \
    && rm -rf /etc/apt/apt.conf.d/docker-clean 
    # # Setup Rosdep
    # && rm /etc/ros/rosdep/sources.list.d/20-default.list \
    # && rosdep init \
    # && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y sudo    
# Add user user and switch to home directory
# Create user with specified UID/GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && usermod -aG sudo $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/90-$USERNAME \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME


# Setup ros
RUN source /opt/ros/${ROS_DISTRO}/setup.sh \
    && rosdep update \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

USER root

# Setup entrypoint
COPY entrypoint.sh /
RUN chmod 0755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Setup message definitions
COPY origin-msgs_arm64_1.0.1.deb /
RUN dpkg -i /origin-msgs_arm64_1.0.1.deb || true \
    && apt-get install -f -y

# Install extra dependencies
# RUN sudo apt update && sudo apt install -y \
#     <package you want to install>

#RUN pip3 install spatial-correlation-sampler==0.4.0

# Switch to the new user
USER $USERNAME
WORKDIR /home/$USERNAME/flow_ws
# Copy workspace into container
COPY flow_ws/ /home/$USERNAME/flow_ws/

# Fix permissions
COPY --chown=zuleikarg:zuleikarg flow_ws/ /home/zuleikarg/flow_ws/

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /home/$USERNAME/flow_ws && \
    colcon build

CMD ["/bin/bash"]

