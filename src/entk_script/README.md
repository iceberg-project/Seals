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

__Note: GSISSH and My-Proxy installation is mainly tested for Ubuntu 16.04. In case you have some other distribution please try to use the commented instructions.__

In addition, you need a MongoDB either installed in your Virtual Machine ([instructions through Docker](https://codehangar.io/mongodb-image-instance-with-docker-toolbox-tutorial/)) 
or you can use a Mongo as a Service via [MLab.com](https://mlab.com/). 

If you installed your own MongoDB, note somewhere the followin URL:
`mongodb://<ip_to _VM>:<mongodb_port>/entk_db`

Mlad provides you with a similar URL. Make sure your DB is password protected.

Now we are ready to install Ensemble Toolkit. The instructions will be based on installation from PyPi and Conda

### PyPi installation:

```
virtualenv rp
source rp/bin/activate
pip install radical.entk
```

### Conda Installation:

```
conda create -y -p entk_env python=2.7 radical.pilot -c conda-forge
source activate entk_env
conda install radical.entk
```

After the installation has finished, please run the following:
```
  python               : 2.7.15
  pythonpath           :
  virtualenv           : entk_env

  radical.entk         : 0.7.6
  radical.pilot        : 0.50.8
  radical.utils        : 0.50.2
  saga                 : 0.50.0
```
The version numbers may be different, but the overall style should not.

## Execution
Initially export the following

```
export RADICAL_PILOT_DBURL=mongodb://<dbuser>:<dbpassword>@ds125872.mlab.com:25872/re_rp_devel
```

Create a proxy with XSEDE for 11 days

```
 myproxy-logon -s myproxy.xsede.org -l <xsede_username> -t 10000
```

You will get a prompt asking: `Enter MyProxy pass phrase`. This would be the password 
you have at the XSEDE portal. Success will provide you with a credentials file under
`/tmp`.

For example:
```
iparask@DESKTOP-R64I4QR:~$ myproxy-logon -l iparask -s myproxy.xsede.org -t 72
Enter MyProxy pass phrase:
A credential has been received for user iparask in /tmp/x509up_u1000.
iparask@DESKTOP-R64I4QR:~$
```

To verify that your certificate is valid, as well as, its remaining time do:
```
iparask@DESKTOP-R64I4QR:~$ grid-proxy-info
subject  : /C=US/O=National Center for Supercomputing Applications/CN=Ioannis Paraskevakos
issuer   : /C=US/O=National Center for Supercomputing Applications/OU=Certificate Authorities/CN=MyProxy CA 2013
identity : /C=US/O=National Center for Supercomputing Applications/CN=Ioannis Paraskevakos
type     : end entity credential
strength : 2048 bits
path     : /tmp/x509up_u1000
timeleft : 263:59:22  (11.0 days)
```

And now do
```
python entk_script.py cpus gpus queue images
```
For additional information about the scripts arguments execute:
```
python entk_script.py -h
```
