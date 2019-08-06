#!/usr/bin/env python

__author__    = 'ICEBERG CI Team'
__email__     = 'iceberg-ci@googlegroups.com'
__copyright__ = 'Copyright 2018-19'
__license__   = 'MIT'


""" Setup script, only usable via pip. """

import re
import os
import sys
import glob
import shutil
import subprocess as sp

name     = 'iceberg_seals.search'
mod_root = 'src/'

try:
    from setuptools import setup, Command, find_packages
except ImportError as e:
    print("%s needs setuptools to install" % name)
    sys.exit(1)


# ------------------------------------------------------------------------------
#
# versioning mechanism:
#
#   - version:          1.2.3            - is used for installation
#   - version_detail:  v1.2.3-9-g0684b06 - is used for debugging
#   - version is read from VERSION file in src_root, which then is copied to
#     module dir, and is getting installed from there.
#   - version_detail is derived from the git tag, and only available when
#     installed from git.  That is stored in mod_root/VERSION in the install
#     tree.
#   - The VERSION file is used to provide the runtime version information.
#
def get_version(mod_root):
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

        version_detail = p.communicate()[0]
        version_detail = version_detail.decode('utf-8')
        version_detail = version_detail.replace('detached from ', 'detached-')

        # remove all non-alphanumeric (and then some) chars
        version_detail = re.sub('[/ ]+', '-', version_detail)
        version_detail = re.sub('[^a-zA-Z0-9_+@.-]+', '', version_detail)

        if  p.returncode   !=  0  or \
            version_detail == '@' or \
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
if  sys.hexversion < 0x03050000:
    raise RuntimeError("%s requires Python 3.x (3.5 or higher)" % name)


# ------------------------------------------------------------------------------
# get version info -- this will create VERSION and srcroot/VERSION
version, version_detail, sdist_name = get_version(mod_root)


# ------------------------------------------------------------------------------
class our_test(Command):
    user_options = []
    def initialize_options(self): pass
    def finalize_options  (self): pass
    def run(self):
        testdir = "%s/tests/" % os.path.dirname(os.path.realpath(__file__))
        retval  = sp.call([sys.executable,
                          '%s/run_tests.py'          % testdir,
                          '%s/configs/basetests.cfg' % testdir])
        raise SystemExit(retval)


# ------------------------------------------------------------------------------
#
def read(fname):
    try :
        return open(fname).read()
    except Exception :
        return ''

# -------------------------------------------------------------------------------
setup_args = {
    'name'               : name,
    'version'            : version,
    'description'        : 'The ICEBERG Seals usecase package',
    'long_description'   : (read('README.md') + '\n\n' + read('CHANGES.md')),
    'long_description_content_type' : "text/markdown",
    'author'             : 'ICEBERG Team',
    'author_email'       : 'iceberg-ci@googlegroups.com',
    'url'                : 'https://www.github.com/iceberg-project/seals/',
    'license'            : 'MIT',
    'classifiers'        : [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix'
    ],
    'namespace_packages': ['iceberg_seals','iceberg_seals.search'],
    'packages'          : find_packages('src/'),

    'package_dir'       : {'': 'src/'},

    'package_data'      :  {'': ['VERSION', 'SDIST', sdist_name]},

    'install_requires'  :  ['numpy>1.12',
                            'scipy',
                            'torch',
                            'torchvision',
                            'tensorboardX',
                            'opencv-python',
                            'rasterio',
                            'affine',
                            'geopandas',
                            'pandas'
                           ],
    'zip_safe'          : False,
    'scripts'           : ['bin/iceberg_seals.tiling',
                           'bin/iceberg_seals.predicting'
                           ]
}

# ------------------------------------------------------------------------------

setup(**setup_args)

# ------------------------------------------------------------------------------
