"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
# pylint: disable=protected-access, unused-argument, unused-import

import os
import random
import mock
import pandas as pd
from src.entk_script.image_disc import image_discovery

@mock.patch('src.entk_script.image_disc.glob',
            return_value=['test1.tif', 'test2.tif'])
def test_image_discovery_1(mocked_glob):
    """
    Test simple CSV contents
    """

    expected_values = pd.DataFrame(columns=['Filename'],
                                   data=['test1.tif', 'test2.tif'])
    # Test just the CSV file contents
    image_discovery(path='./')

    test = pd.read_csv('list.csv')

    assert test.equals(expected_values)

    os.remove('list.csv')


def test_image_discovery_2():
    """
    Test CSV Filename
    """

    # Test just the CSV file name
    image_discovery(path='./', filename='test.csv')

    assert os.path.isfile('test.csv')

    os.remove('test.csv')

@mock.patch('src.entk_script.image_disc.glob',
            return_value=['test1.tif', 'test2.tif'])
@mock.patch('src.entk_script.image_disc.os.path.getsize',
            side_effect=[1048576, 4194304])
def test_image_discovery_3(mocked_glob, mocked_getsize):
    """
    Test CSV Filesizes
    """

    expected_values = pd.DataFrame(columns=['Filename', 'Size'],
                                   data=[['test1.tif', 1], ['test2.tif', 4]])
    # Test just the CSV file contents
    image_discovery(path='./', filesize=True)

    test = pd.read_csv('list.csv')
    print test
    assert test.equals(expected_values)

    os.remove('list.csv')
