from radical.entk import Pipeline, Stage, Task, AppManager, ResourceManager

def generate_pipeline(name, stages):  #generate the pipeline of prediction and blob detection

    # Create a Pipeline object
    p = Pipeline()
    p.name = name

    for s_cnt in range(stages):


        if(s_cnt==0):
            # Create a Stage object
            s0 = Stage()
            s0.name = 'Stage %s'%s_cnt
            # Create Task 1, training
            t1 = Task()
            t1.name = 'Predictor'
            t1.pre_exec = ['module load psc_path/1.1',
                           'module load slurm/default',
                           'module load intel/17.4',
                           'module load python3',
                           'module load cuda',
                           'mkdir -p classified_images/crabeater',
                           'mkdir -p classified_images/weddel',
                           'mkdir -p classified_images/pack-ice',
                           'mkdir -p classified_images/other',
                           'source /pylon5/mc3bggp/paraskev/pytorchCuda/bin/activate'
                          ]
            t1.executable = 'python3'   # Assign executable to the task   
            # Assign arguments for the task executable
            t1.arguments = ['pt_predict.py','-class_names','crabeater','weddel','pack-ice','other']
            t1.link_input_data = ['/pylon5/mc3bggp/paraskev/seal_test/nn_model.pth.tar',
                                  '/pylon5/mc3bggp/paraskev/nn_images',
                                  '/pylon5/mc3bggp/paraskev/seal_test/test_images'
                                  ]
            t1.upload_input_data = ['pt_predict.py','sealnet_nas_scalable.py']
            t1.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
            t1.gpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
        
            s0.add_tasks(t1)    
            # Add Stage to the Pipeline
            p.add_stages(s0)
        else:
            # Create a Stage object
            s1 = Stage()
            s1.name = 'Stage %s'%s_cnt
            # Create Task 2,
            t2 = Task()
            t2.pre_exec = ['module load psc_path/1.1',
                           'module load slurm/default',
                           'module load intel/17.4',
                           'module load python3',
                           'module load cuda',
                           'module load opencv',
                           'source /pylon5/mc3bggp/paraskev/pytorchCuda/bin/activate',
                           'mkdir -p blob_detected'
                         ]
            t2.name = 'Blob_detector'         
            t2.executable = ['python3']   # Assign executable to the task   
            # Assign arguments for the task executable
            t2.arguments = ['blob_detector.py']
            t2.upload_input_data = ['blob_detector.py']
            t2.link_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/classified_images'%(p.uid, s0.uid, t1.uid)]
            t2.download_output_data = ['blob_detected/'] #Download resuting images 
            t2.cpu_reqs = {'processes': 1,'threads_per_process': 1, 'thread_type': 'OpenMP'}
            t2.gpu_reqs = {'processes': 1, 'threads_per_process': 1, 'thread_type': 'OpenMP'}
            s1.add_tasks(t2)
            # Add Stage to the Pipeline
            p.add_stages(s1)

    return p


if __name__=='__main__':
    p1 = generate_pipeline(name='Pipeline 1', stages=2)
    
    res_dict = {'resource': 'xsede.bridges',
             'walltime': 30,
             'cpus': 12,
             'gpus': 2,
             'schema' : 'gsisshh',
             'project': 'mc3bggp',
             'queue' : 'GPU-small'
    }
    
    
    # Create Resource Manager
    rman = ResourceManager(res_dict)

    # Create Application Manager
    appman = AppManager(port=32773)
    
    # Assign resource manager to the Application Manager
    appman.resource_manager = rman
    
    # Assign the workflow as a set of Pipelines to the Application Manager
    appman.assign_workflow(set([p1]))

    # Run the Application Manager
    appman.run()
