# Introduction

This README file contains basic instructions on how to execute the Ensemble Toolkit 
script for this use case

It is suggested to read [RADICAL-Pilot's documentation](https://radicalpilot.readthedocs.io/en/latest/)
and [Ensemble Toolkit's documentation](https://radicalentk.readthedocs.io/en/latest/).

It is also recommended that the execution is done from a Virtual Machine that has constant 
Internet connection, Docker, and runs some version of Ubuntu.

## Preparing your environment for installation

Verify that you have a Python 2.7 installation and you can create a virtual environment
by executing `virtualenv` or `conda create -n test`. Please do not create a virtual environment now.

If you are not sure, please install [Miniconda 2](https://conda.io/miniconda.html) before proceeding.
In either case, instructions are provided for both GCC python and Conda Python.

The next step would be to install Rabbit MQ, since it is needed from Ensemble Toolkit:

```
~$ docker run -d --name entk_queues -P rabbitmq:3
~$ docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                                                                                                 NAMES
d334b23418f2        rabbitmq:3          "docker-entrypoint..."   3 months ago        Up 3 months         0.0.0.0:32775->4369/tcp, 0.0.0.0:32774->5671/tcp, 0.0.0.0:32773->5672/tcp, 0.0.0.0:32772->25672/tcp   entk_queues
```
Verify that there is a port that looks like `0.0.0.0:32772->25672/tcp`. The second part should be the same.

After RabbitMQ is installed, we need to install gsissh and my-proxy. Instructions
can be found [here](https://github.com/vivek-bala/docs/blob/master/misc/gsissh_setup_stampede_ubuntu_xenial.sh)
Please change xenial with your Ubuntu codename (to find out run: `lsb_release -a`).

In addition, you need a MongoDB either installed in your Virtual Machine ([instructions through Docker](https://codehangar.io/mongodb-image-instance-with-docker-toolbox-tutorial/)) 
or you can use a Mongo as a Service via [MLab.com](https://mlab.com/)

Now we are ready to install Ensemble Toolkit. The instructions will be based on installation from PyPi and Conda

### PyPi installation:

```
vitrualenv rp
source rp/bin/activate
pip install radical.entk
```

### Conda Installation:

```
conda create -y -p entk_env python=2.7 radical.pilot -c conda-forge
source activate entk_env
pip install radical.entk
```

## Execution
Initially export the following

```
export RADICAL_PILOT_DBURL=<mongodburl>
```

Create a proxy with XSEDE for 72 hours

```
 myproxy-logon -s myproxy.xsede.org -l <username> -t 72
```

And now do
```
python entk_script.py cpus gpus queue images
```
For additional information about the scripts arguments execute:
```
python entk_script.py -h
```