from radical.entk import Pipeline,Stage,Task,ResourceManager,AppManager
import argparse

def generate_pipeline(name,stages,image,tile_size,
                      pipeline,model_arch,model_name,hyperparam_set,dev):  #generate the pipeline of prediction and blob detection

    # Create a Pipeline object
    p = Pipeline()
    p.name = name
    for s_cnt in range(stages):


        if(s_cnt==0):
            # Create a Stage object
            s0 = Stage()
            s0.name = 'Stage %s'%s_cnt
            # Create Task 1, training
            t0 = Task()
            t0.name = 'Tiling'
            t0.pre_exec = ['module load psc_path/1.1',
                           'module load slurm/default',
                           'module load intel/17.4',
                           'module load python3',
                           'source $SCRATCH/pytorchCuda/bin/activate',
                           'hostname'
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
        elif (s_cnt==1):
             # Create a Stage object
            s1 = Stage()
            s1.name = 'Stage %s'%s_cnt
            # Create Task 1, training
            t1 = Task()
            t1.name = 'PredictingCounting'
            t1.pre_exec = ['module load psc_path/1.1',
                           'module load slurm/default',
                           'module load intel/17.4',
                           'module load python3',
                           'module load cuda',
                           'source $SCRATCH/pytorchCuda/bin/activate',
                           'hostname',
                           'export CUDA_VISIBLE_DEVICES=%d' % dev
                          ]
            t1.executable = 'python3'   # Assign executable to the task   
            # Assign arguments for the task executable
            t1.arguments = ['predict_sealnet.py','--pipeline',pipeline,
                            '--dest_folder','./','--test_dir','./','--model_architecture',model_arch,
                           '--hyperparameter_set',hyperparam_set,'--model_name',model_name]
            t1.link_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/tiles'%(p.uid, s0.uid, t0.uid),
                                  '/pylon5/mc3bggp/paraskev/models/%s.tar' % model_name]
            t1.upload_input_data = ['predict_sealnet.py','utils/']
            t1.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
            t1.gpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
        
            s1.add_tasks(t1)    
            # Add Stage to the Pipeline
            p.add_stages(s1)
        else:
            # Create a Stage object
            s2 = Stage()
            s2.name = 'Stage %s'%s_cnt
            # Create Task 2,
            t2 = Task()
            t2.pre_exec = ['module load psc_path/1.1',
                           'module load slurm/default',
                           'module load intel/17.4',
                           'module load python3',
                           'module load cuda',
                           'source $SCRATCH/pytorchCuda/bin/activate'
                          ]
            t2.name = 'AggregateResults'         
            t2.executable = ['python']   # Assign executable to the task   
            # Assign arguments for the task executable
            t2.arguments = ['aggregate_predictions.py','%s_predictions.csv'%model_name]
            t2.upload_input_data = ['aggregate_predictions.py']
            for t in s1.tasks:
                t2.link_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/%s_predictions.csv>%s_predictions.csv'%(p.uid, s1.uid, t.uid,model_name,t.uid)]
            t2.download_output_data = ['%s_predictions.csv> %s_%s_predictions.csv'%(model_name,p.uid,model_name)] #Download resuting images 
            t2.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
            s2.add_tasks(t2)
            # Add Stage to the Pipeline
            p.add_stages(s2)

    return p


if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser(description='Scaling inputs')
    parser.add_argument('cores', type=int, help='Number of Cores')
    parser.add_argument('gpus', type=int, help='Number of GPUs')
    parser.add_argument('queue',type=str, help='Queue to submit to')
    parser.add_argument('images',type=int, help='Number of images to use')
    args = parser.parse_args()
    
    images = [] # a list with images paths on bridges
    
    res_dict = {'resource': 'xsede.bridges',
                'walltime': 30,
                'cpus': args.cores,
                'gpus': args.gpus,
                'schema' : 'gsissh',
                'project': '',
                'queue' : args.queue
               }
    
    
    # Create Resource Manager
    rman = ResourceManager(res_dict)

    # Create Application Manager
    appman = AppManager(port=32773)
    
    # Assign resource manager to the Application Manager
    appman.resource_manager = rman
    pipelines = list()
    dev = 0
    for cnt in range(args.images):
        p1 = generate_pipeline(name = 'Pipeline%s'%cnt,
                               stages = 3,
                               image = images[cnt],
                               tile_size = 299,
                               pipeline = 'Pipeline1.1',
                               model_arch = 'WideResnetCount',
                               model_name = 'WideResnetCount',
                               hyperparam_set = 'A',
                               dev = dev
                               )
        dev = dev ^ 1
        pipelines.append(p1)
      # Assign the workflow as a set of Pipelines to the Application Manager
    appman.assign_workflow(set(pipelines))

    # Run the Application Manager
    appman.run()
