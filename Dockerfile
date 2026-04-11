from ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update && apt -y install \
    gcc git \
    wget vim curl \
    python3-pip cmake automake \
    build-essential ca-certificates \
    apt-utils flex bison mona ccache

# installing spot using apt
# RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -
#RUN wget --no-check-certificate -q -O /etc/apt/keyrings/lre-epita.gpg https://www.lre.epita.fr/repo/debian.gpg

SHELL ["/bin/bash", "-c"] 

RUN pip3 install pyyaml numpy bidict networkx graphviz ply pybullet pyperplan==1.3 cython IPython svgwrite matplotlib imageio lark-parser==0.9.0 sympy==1.6.1

# RUN echo 'deb https://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list
# RUN mkdir -p -m 0755 /etc/apt/keyrings
# RUN echo "deb [signed-by=/etc/apt/keyrings/lre-epita.gpg] http://www.lre.epita.fr/repo/debian/ stable/" \
#  > /etc/apt/sources.list.d/lre-epita.list


# try --allow-unauthenticated if authentication fails
#RUN apt-get -y update && apt-get -y install spot \
#    libspot-dev \
#    spot-doc

# Add the ROS GPG key and repository
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc && \
    apt-key add ros.asc && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'

# Update the package list again
RUN apt-get update

# Install ROS Noetic and rosdep
RUN apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    ros-noetic-moveit \
    ros-noetic-moveit-visual-tools \
    ros-noetic-moveit-planners-ompl \
    ros-noetic-moveit-ros-planning-interface \
    ros-noetic-moveit-commander \
    ros-noetic-moveit-simple-controller-manager \
    ros-noetic-moveit-ros-control-interface \
    ros-noetic-moveit-ros-visualization \
    ros-noetic-vrpn-client-ros \
    ros-noetic-franka-control \
    ros-noetic-panda-moveit-config


# Install required libraries for Qt X11
RUN apt-get update && apt-get install -y \
    libxcb-xinerama0 \
    libqt5gui5 \
    libqt5widgets5 \
    x11-apps


# Initialize rosdep
RUN rosdep init && \
    rosdep update

# install eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
RUN tar -xzvf eigen-3.4.0.tar.gz
RUN cd eigen-3.4.0 && mkdir build && cd build
WORKDIR /eigen-3.4.0/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
RUN make && make install

# Clean up unnecessary ROS files
RUN rm -rf ros.asc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH="/usr/local/bin:${PATH}"
ENV ROS_DISTRO noetic

#### SETUP RELATED TO PLANNER
# installing system packages
ADD cudd_and_wrapper/cudd /cudd
ADD cudd_and_wrapper/PyDD /PyDD

ADD ./ /root/symbolic_planning/src

RUN rm -rf /root/symbolic_planning/src/cudd && rm -rf /root/symbolic_planning/src/PyDD


#### SETUP RELATED TO SIMULATOR ####

# Create a catkin workspace
RUN mkdir -p /root/ws/src
WORKDIR /root/ws

RUN rosdep update
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"

RUN /bin/bash -c "echo 'export PATH=/usr/lib/ccache:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc"

WORKDIR /root/ws/src
RUN git clone https://github.com/peteramorese/taskit.git

#### INSTALLATION RELATED TO SIMULATOR ####
WORKDIR /root/ws
RUN catkin config --install --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release 

# Build the workspace
RUN catkin build
RUN echo 'source ~/ws/devel/setup.bash' >> ~/.bashrc

#### INSTALLATION RELATED TO PLANNER ####
RUN cd /cudd/ && \
    ./configure --enable-shared \
    --enable-obj --enable-dddmp \
    && make -j4 \
    && make check && make install  

RUN cd /PyDD && python3 setup.py install

RUN cd /root/symbolic_planning/src/src/LTLf2DFA && pip3 install .

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc

# SPOT Related stuff
# Download, Extract, Build, Install, and CLEAN UP
RUN wget -q http://www.lre.epita.fr/dload/spot/spot-2.14.5.tar.gz \
    && tar -xzf spot-2.14.5.tar.gz \
    && cd spot-2.14.5 \
    && ./configure \
    && make -j16 \
    && make install \
    && cd .. \
    && rm -rf spot-2.14.5 spot-2.14.5.tar.gz

#### Change working directory ####
WORKDIR /root/symbolic_planning/src

# make plots directory where all the plots are dumped
RUN mkdir -p plots
RUN pip3 install tabulate
ENTRYPOINT ["/bin/bash", "-c"]
