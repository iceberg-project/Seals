from glob import glob
from radical.entk import Pipeline,Stage,Task,AppManager
import argparse
import pandas as pd

def generate_discover_pipeline(path):
    '''
    This function takes as an input a path on Bridges and returns a pipeline that
    will provide a file for all the images that exist in that path.
    '''
    p = Pipeline()
    p.name = 'Disc'
    s = Stage()
    s.name = 'Disc-S0'
    # Create Task 1, training
    t0 = Task()
    t0.name = 'Disc-T0'
    t0.pre_exec = ['module load psc_path/1.1',
                   'module load slurm/default',
                   'module load intel/17.4',
                   'module load python3',
                   'source $SCRATCH/pytorchCuda/bin/activate'
                  ]
    t0.executable = 'python3'   # Assign executable to the task   
    t0.arguments = ['image_disc.py','%s'%path,'--filename=images.csv','--filesize']
    t0.download_output_data = ['images.csv']
    t0.upload_input_data = ['image_disc.py']
    t0.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
    s.add_tasks(t0)
    # Add Stage to the Pipeline
    p.add_stages(s)

    return p

def generate_pipeline(name,image,tile_size,pipeline,model_path,model_arch,
                      model_name,hyperparam_set,dev,output_dir):  #generate the pipeline of prediction and blob detection

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
        :dev: Which GPU device will be used by this pipeline, int
    '''
    # Create a Pipeline object
    p = Pipeline()
    p.name = name
    # Create a Stage object
    s0 = Stage()
    s0.name = '%s-S0' % (name)
    # Create Task 1, training
    t0 = Task()
    t0.name = '%s-T0' % s0.name
    t0.pre_exec = ['module load psc_path/1.1',
                   'module load slurm/default',
                   'module load intel/17.4',
                   'module load python3',
                   'source $SCRATCH/pytorchCuda/bin/activate'
                  ]
    t0.executable = 'python3'   # Assign executable to the task   
    # Assign arguments for the task executable
    t0.arguments = ['tile_raster.py','--scale_bands=%s'%tile_size,'--input_image=%s'%image.split('/')[-1]]
    t0.link_input_data = [image]
    t0.upload_input_data = ['tile_raster.py']
    t0.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}

    s0.add_tasks(t0)
    # Add Stage to the Pipeline
    p.add_stages(s0)
    
    # Create a Stage object
    s1 = Stage()
    s1.name = '%s-S1' % (name)
    # Create Task 1, training
    t1 = Task()
    t1.name = '%s-T1' % s1.name
    t1.pre_exec = ['module load psc_path/1.1',
                   'module load slurm/default',
                   'module load intel/17.4',
                   'module load python3',
                   'module load cuda',
                   'source $SCRATCH/pytorchCuda/bin/activate',
                   'export CUDA_VISIBLE_DEVICES=%d' % dev
                  ]
    t1.executable = 'python3'   # Assign executable to the task   
    # Assign arguments for the task executable
    t1.arguments = ['predict_sealnet.py','--pipeline',pipeline,
                    '--dest_folder','./','--test_dir','./','--model_architecture',model_arch,
                   '--hyperparameter_set',hyperparam_set,'--model_name',model_name]
    t1.link_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/tiles'%(p.name, s0.name, t0.name),
                          '/pylon5/mc3bggp/paraskev/models/%s.tar' % model_name]
    t1.upload_input_data = ['predict_sealnet.py','utils/']
    t1.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t1.gpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t1.download_output_data = ['%s_predictions.csv> %s/%s_predictions.csv'%(model_name,output_dir,image.split('/')[-1])] #Download resuting images 

    s1.add_tasks(t1)    
    # Add Stage to the Pipeline
    p.add_stages(s1)
        
    return p

def create_aggregated_output(images,path):
    
    '''
    This function takes a list of images and aggregates the results into a single CSV file
    '''

    aggregated_results = pd.DataFrame(columns=['Image','Seals'])
    for image in images:
        image_pred = pd.read_csv(path+image.split('/')[-1]+'_predictions.csv')
        aggregated_results.loc[len(aggregated_results)] = [image.split('/')[-1],image_pred['predictions'].sum()]

    aggregated_results.to_csv(path+'/seal_predictions.csv',index=False)


def args_parser():
    parser = argparse.ArgumentParser(description='Executes the Seals pipeline for a set of images')
    
    parser.add_argument('-c', '--cpus', type=int, default=1, help='The number of CPUs required for execution')
    parser.add_argument('-g', '--gpus', type=int, default=1, help='The number of GPUs required for execution')
    parser.add_argument('-ip','--input_dir', type=str,help='Images inpuit directory on the selected resource')
    parser.add_argument('-m', '--model', type=str,help='Which model will be used')
    parser.add_argument('-op','--output_dir', type=str,help='Path to folder that the output will be stored')
    parser.add_argument('-p', '--project',type=str,help='The project that will be charged')
    parser.add_argument('-q', '--queue',type=str,help='The queue from which we request resources.')
    parser.add_argument('-r', '--resource', type=str,help='HPC resource whit script will run.')
    parser.add_argument('-w', '--walltime', type=int, help='The amount of time resources are requested')

    return parser.parse_args()

if __name__=='__main__':
    
    args = args_parser()
    
   
    res_dict = {'resource': args.resource,
                'walltime': args.walltime,
                'cpus': args.cpus,
                'gpus': args.gpus,
                'schema' : 'gsissh',
                'project': args.project,
                'queue' : args.queue
               }
    try:

        # Create Application Manager
        appman = AppManager(port=32773,hostname='localhost',autoterminate=False)
    
        # Assign resource manager to the Application Manager
        appman.resource_desc = res_dict

        #Create a task that discovers the dataset
        disc_pipeline = generate_discover_pipeline(args.input_dir)
        appman.workflow = set([disc_pipeline])
        
        # Run
        appman.run()

        images = pd.read_csv('images.csv')['Filename'].tolist()
        
        # Create a single pipeline per image
        pipelines = list()
        dev = 0
        for cnt in range(len(images)):
            p1 = generate_pipeline(name = 'P%s'%cnt,
                               image = images[cnt],
                               tile_size = 299,
                               pipeline = 'Pipeline1.1',
                               model_path = args.model,
                               model_arch = 'WideResnetCount',
                               model_name = 'WideResnetCount',
                               hyperparam_set = 'A',
                               dev = dev,
                               output_dir = args.output_dir
                               )
            dev = dev ^ 1
            pipelines.append(p1)
        # Assign the workflow as a set of Pipelines to the Application Manager
        appman.workflow = set(pipelines)

        # Run the Application Manager
        appman.run()
        
        # Get all results and produce a single one.
        create_aggregated_output(images,args.output_dir)

    finally:
        # Now that all images have been analyzed, release the resources.
        appman.resource_terminate()

    