__author__    = "Ioannis Paraskevakos"
__copyright__ = "Copyright 2019. The ICEBERG Project"
__license__   = "MIT"

import os
import sys
import subprocess as sp
import re
import shutil

name = 'iceberg.seals'
mod_root = 'iceberg/seals'

try:
    from setuptools import setup, Command, find_packages
except ImportError as e:
    print("%s needs setuptools to install" % name)
    sys.exit(1)

def set_version(mod_root):
    """
    mod_root
        a VERSION file containes the version strings is created in mod_root,
        during installation.  That file is used at runtime to get the version
        information.
        """

    try:

        version_base   = None
        version_detail = None

        # get version from './VERSION'
        src_root = os.path.dirname(__file__)
        if  not src_root:
            src_root = '.'

        with open(src_root + '/VERSION', 'r') as f:
            version_base = f.readline().strip()

        # attempt to get version detail information from git
        # We only do that though if we are in a repo root dir,
        # ie. if 'git rev-parse --show-prefix' returns an empty string --
        # otherwise we get confused if the ve lives beneath another repository,
        # and the pip version used uses an install tmp dir in the ve space
        # instead of /tmp (which seems to happen with some pip/setuptools
        # versions).
        p = sp.Popen('cd %s ; '
                     'test -z `git rev-parse --show-prefix` || exit -1; '
                     'tag=`git describe --tags --always` 2>/dev/null ; '
                     'branch=`git branch | grep -e "^*" | cut -f 2- -d " "` 2>/dev/null ; '
                     'echo $tag@$branch' % src_root,
                     stdout=sp.PIPE, stderr=sp.STDOUT, shell=True)
        version_detail = str(p.communicate()[0].strip())
        version_detail = version_detail.replace('detached from ', 'detached-')

        # remove all non-alphanumeric (and then some) chars
        version_detail = re.sub('[/ ]+', '-', version_detail)
        version_detail = re.sub('[^a-zA-Z0-9_+@.-]+', '', version_detail)

        if  p.returncode   !=  0  or \
            version_detail == '@' or \
            'git-error' in version_detail or \
            'not-a-git-repo' in version_detail or \
            'not-found'      in version_detail or \
            'fatal'          in version_detail :
            version = version_base
        elif '@' not in version_base:
            version = '%s-%s' % (version_base, version_detail)
        else:
            version = version_base

        # make sure the version files exist for the runtime version inspection
        path = '%s/%s' % (src_root, mod_root)
        with open(path + "/VERSION", "w") as f:
            f.write(version + "\n")

        sdist_name = "%s-%s.tar.gz" % (name, version)
        sdist_name = sdist_name.replace('/', '-')
        sdist_name = sdist_name.replace('@', '-')
        sdist_name = sdist_name.replace('#', '-')
        sdist_name = sdist_name.replace('_', '-')

        if '--record'    in sys.argv or \
           'bdist_egg'   in sys.argv or \
           'bdist_wheel' in sys.argv    :
          # pip install stage 2 or easy_install stage 1
          #
          # pip install will untar the sdist in a tmp tree.  In that tmp
          # tree, we won't be able to derive git version tags -- so we pack the
          # formerly derived version as ./VERSION
            shutil.move("VERSION", "VERSION.bak")            # backup version
            shutil.copy("%s/VERSION" % path, "VERSION")      # use full version instead
            os.system  ("python setup.py sdist")             # build sdist
            shutil.copy('dist/%s' % sdist_name,
                        '%s/%s'   % (mod_root, sdist_name))  # copy into tree
            shutil.move("VERSION.bak", "VERSION")            # restore version

        with open(path + "/SDIST", "w") as f:
            f.write(sdist_name + "\n")

        return version_base, version_detail, sdist_name

    except Exception as e :
        raise RuntimeError('Could not extract/set version: %s' % e)

# ------------------------------------------------------------------------------
# check python version. we need >= 2.7, <3.x
if  sys.hexversion < 0x02070000 or sys.hexversion >= 0x03000000:
    raise RuntimeError("%s requires Python 2.x (2.7 or higher)" % name)


# ------------------------------------------------------------------------------
# get version info -- this will create VERSION and srcroot/VERSION
version, version_detail, sdist_name = set_version(mod_root)

setup_args = {
    'name'             : name,
    'version'          : version,
    'description'      : "ICEBERG Seals Package.",
    'author'           : 'RADICAL Group at Rutgers University',
    'author_email'     : 'g.paraskev@rutgers.edu',
    'maintainer'       : "Ioannis Paraskevakos",
    'maintainer_email' : 'g.paraskev@rutgers.edu',
    'url'              : 'https://github.com/iceberg-project/Seals/',
    'license'          : 'MIT',
    'keywords'         : "high-resolution imagery workflow execution",
    'classifiers'      :  [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Utilities',
        'Topic :: System :: Distributed Computing',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix'
    ],

    'namespace_packages': ['iceberg.seals', 'iceberg.seals'],
    'packages'          : find_packages(mod_root),

    'package_dir'       : {'': 'iceberg/seals/src'},

    'package_data'      :  {'': ['VERSION', 'SDIST']},

    'install_requires'  :  ['radical.entk', 'pandas'],

    'zip_safe'          : False

}

setup (**setup_args)


'''
To publish to pypi:
python setup.py sdist
twine upload --skip-existing dist/<tarball name>
'''