"""
Seals Use Case EnTK Analysis Script
==========================================================

This script contains the EnTK Pipeline script for the Seals Use Case

Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
import argparse
import pandas as pd

from radical.entk import Pipeline, Stage, Task, AppManager


def generate_discover_pipeline(path):
    '''
    This function takes as an input a path on Bridges and returns a pipeline
    that will provide a file for all the images that exist in that path.
    '''
    pipeline = Pipeline()
    pipeline.name = 'Disc'
    stage = Stage()
    stage.name = 'Disc-S0'
    # Create Task 1, training
    task = Task()
    task.name = 'Disc-T0'
    task.pre_exec = ['module load psc_path/1.1',
                     'module load slurm/default',
                     'module load intel/17.4',
                     'module load python3',
                     'source $SCRATCH/pytorchCuda/bin/activate']
    task.executable = 'python3'   # Assign executable to the task
    task.arguments = ['image_disc.py', '%s' % path, '--filename=images.csv',
                      '--filesize']
    task.download_output_data = ['images.csv']
    task.upload_input_data = ['image_disc.py']
    task.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                     'thread_type': 'OpenMP'}
    stage.add_tasks(task)
    # Add Stage to the Pipeline
    pipeline.add_stages(stage)

    return pipeline


def generate_pipeline(name, image, tile_size, pipeline, model_path, model_arch,
                      model_name, hyperparam_set, device, output_dir):

    '''
    This function creates a pipeline for an image that will be analyzed.

    :Arguments:
        :name: Pipeline name, str
        :image: image path, str
        :tile_size: The size of each tile, int
        :pipeline: Prediction Pipeline, str
        :model_path: Path to the model file, str
        :model_arch: Prediction Model Architecture, str
        :model_name: Prediction Model Name, str
        :hyperparam_set: Which hyperparameter set to use, str
        :device: Which GPU device will be used by this pipeline, int
        :output_dir: The directory in which the output will be stored
    '''
    # Create a Pipeline object
    entk_pipeline = Pipeline()
    entk_pipeline.name = name
    # Create a Stage object
    stage0 = Stage()
    stage0.name = '%s-S0' % (name)
    # Create Task 1, training
    task0 = Task()
    task0.name = '%s-T0' % stage0.name
    task0.pre_exec = ['module load psc_path/1.1',
                      'module load slurm/default',
                      'module load intel/17.4',
                      'module load python3',
                      'source $SCRATCH/pytorchCuda/bin/activate']
    task0.executable = 'python3'   # Assign executable to the task
    # Assign arguments for the task executable
    task0.arguments = ['tile_raster.py', '--scale_bands=%s' % tile_size,
                       '--input_image=%s' % image.split('/')[-1]]
    task0.link_input_data = [image]
    task0.upload_input_data = ['tile_raster.py']
    task0.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}

    stage0.add_tasks(task0)
    # Add Stage to the Pipeline
    entk_pipeline.add_stages(stage0)

    # Create a Stage object
    stage1 = Stage()
    stage1.name = '%s-S1' % (name)
    # Create Task 1, training
    task1 = Task()
    task1.name = '%s-T1' % stage1.name
    task1.pre_exec = ['module load psc_path/1.1',
                      'module load slurm/default',
                      'module load intel/17.4',
                      'module load python3',
                      'module load cuda',
                      'source $SCRATCH/pytorchCuda/bin/activate',
                      'export CUDA_VISIBLE_DEVICES=%d' % device]
    task1.executable = 'python3'   # Assign executable to the task

    # Assign arguments for the task executable
    task1.arguments = ['predict_sealnet.py', '--pipeline', pipeline,
                       '--dest_folder', './', '--test_dir', './',
                       '--model_architecture', model_arch,
                       '--hyperparameter_set', hyperparam_set, '--model_name',
                       model_name]
    task1.link_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/tiles' %
                             (entk_pipeline.name, stage0.name, task0.name),
                             model_path + '/%s.tar' % model_name]
    task1.upload_input_data = ['predict_sealnet.py', 'utils/']
    task1.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}
    task1.gpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}
    # Download resuting images
    task1.download_output_data = ['%s_predictions.csv> %s/%s_predictions.csv' %
                                  (model_name, output_dir,
                                   image.split('/')[-1])]

    stage1.add_tasks(task1)
    # Add Stage to the Pipeline
    entk_pipeline.add_stages(stage1)

    return entk_pipeline


def create_aggregated_output(image_names, path):

    '''
    This function takes a list of images and aggregates the results into a
    single CSV file
    '''

    aggr_results = pd.DataFrame(columns=['Image', 'Seals'])
    for image in image_names:
        image_pred = pd.read_csv(path + image.split('/')[-1] +
                                 '_predictions.csv')
        aggr_results.loc[len(aggr_results)] = [image.split('/')[-1],
                                               image_pred['predictions'].sum()]

    aggr_results.to_csv(path + '/seal_predictions.csv', index=False)


def args_parser():

    '''
    Argument Parsing Function for the script.
    '''
    parser = argparse.ArgumentParser(description='Executes the Seals pipeline\
                                                  for a set of images')

    parser.add_argument('-c', '--cpus', type=int, default=1, help='The number \
                        of CPUs required for execution')
    parser.add_argument('-g', '--gpus', type=int, default=1, help='The number \
                        of GPUs required for execution')
    parser.add_argument('-ip', '--input_dir', type=str, help='Images input \
                        directory on the selected resource')
    parser.add_argument('-m', '--model', type=str, help='Which model will be \
                        used')
    parser.add_argument('-op', '--output_dir', type=str, help='Path to folder \
                        that the output will be stored')
    parser.add_argument('-p', '--project', type=str, help='The project that \
                        will be charged')
    parser.add_argument('-q', '--queue', type=str, help='The queue from which \
                        resources are requested.')
    parser.add_argument('-r', '--resource', type=str, help='HPC resource on \
                        which the script will run.')
    parser.add_argument('-w', '--walltime', type=int, help='The amount of \
                        time resources are requested')

    return parser.parse_args()


if __name__ == '__main__':

    args = args_parser()

    res_dict = {'resource': args.resource,
                'walltime': args.walltime,
                'cpus': args.cpus,
                'gpus': args.gpus,
                'schema': 'gsissh',
                'project': args.project,
                'queue': args.queue}

    try:

        # Create Application Manager
        appman = AppManager(port=32773, hostname='localhost',
                            autoterminate=False)

        # Assign resource manager to the Application Manager
        appman.resource_desc = res_dict

        # Create a task that discovers the dataset
        disc_pipeline = generate_discover_pipeline(args.input_dir)
        appman.workflow = set([disc_pipeline])

        # Run
        appman.run()

        images = pd.read_csv('images.csv')['Filename'].tolist()

        # Create a single pipeline per image
        pipelines = list()
        dev = 0
        for idx, image_path in enumerate(images):
            p1 = generate_pipeline(name='P%s' % idx,
                                   image=image_path,
                                   tile_size=299,
                                   pipeline='Pipeline1.1',
                                   model_path=args.model,
                                   model_arch='WideResnetCount',
                                   model_name='WideResnetCount',
                                   hyperparam_set='A',
                                   device=dev,
                                   output_dir=args.output_dir)
            dev = dev ^ 1
            pipelines.append(p1)
        # Assign the workflow as a set of Pipelines to the Application Manager
        appman.workflow = set(pipelines)

        # Run the Application Manager
        appman.run()

        # Get all results and produce a single one.
        create_aggregated_output(images, args.output_dir)

    finally:
        # Now that all images have been analyzed, release the resources.
        appman.resource_terminate()
