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
import radical.utils as ru
from radical.entk.utils import write_workflow
from src.entk_script.entk_script import generate_discover_pipeline, generate_pipeline

def test_generate_discover_pipeline():
    """
    Test if the discover pipeline is correct.
    """
    pass
    
    #pipe_obj = generate_discover_pipeline(path='test')
    #pipeline = write_workflow([pipe_obj],'./',fwrite=False)
    #expected_pipeline = ru.read_json('src/tests/disc_workflow.json')
    #assert pipeline == expected_pipeline


def test_generate_pipeline():
    """
    Test the image analysis pipeline is correct
    """
    pass

    #pipe_obj = generate_pipeline(name='test',
    #                             image='test',
    #                             image_size=500,
    #                             scale_bands=299,
    #                             model_arch='test',
    #                             training_set='default',
    #                             model_name='test',
    #                             hyperparam_set='A',
    #                             device=0)
    #pipeline = write_workflow([pipe_obj],'./',fwrite=False)
    #expected_pipeline = ru.read_json('src/tests/image_workflow.json')
    #assert pipeline == expected_pipeline
