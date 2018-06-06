import argparse
import time

import radical.pilot as rp
import radical.utils as ru

if __name__=='__main__':
    
    report = ru.LogReporter(name='radical.pilot')
    report.title('Getting Started (RP version %s)' % rp.version)
    
    session = rp.Session()
    
    try:
        pdesc = rp.ComputePilotDescription()
        
        pdesc.resource = 'xsede.bridges'
        pdesc.access_schema = 'gsissh'
        pdesc.project = 'mc3bggp'
        pdesc.gpus = 2
        pdesc.cores = 24
        pdesc.runtime = 60
        pdesc.exit_on_error = True
        pdesc.queue = 'GPU'
        
        pmgr = rp.PilotManager(session=session)
        
        pilot = pmgr.submit_pilots(pdesc)
        
        umgr = rp.UnitManager(session=session)
        
        umgr.add_pilots(pilot)
        
        
        cud1 = rp.ComputeUnitDescription()
        
        cud1.executable    = 'python3'
        cud1.gpu_processes = 1
        cud1.cpu_processes = 1
        cud1.pre_exec      = ['module load psc_path/1.1',
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
        cud1.arguments = ['pt_predict.py','-class_names','crabeater','weddel','pack-ice','other']
        cud1.input_staging = [{'source': '/pylon5/mc3bggp/paraskev/seal_test/nn_model.pth.tar',
                               'target': 'unit:///nn_model.pth.tar',
                               'action': rp.LINK},
                               {'source' : '/pylon5/mc3bggp/paraskev/seal_test/test_images',
                                'target' : 'unit:///test_images',
                                'action' : rp.LINK},
                               {'source' : '/pylon5/mc3bggp/paraskev/nn_images',
                                'target' : 'unit:///nn_images',
                                'action' : rp.LINK},
                               {'source' : 'client:///pt_predict.py',
                                'target' : 'unit:///pt_predict.py',
                                'action' : rp.TRANSFER},
                               {'source' : 'client:///sealnet_nas_scalable.py',
                                'target' : 'unit:///sealnet_nas_scalable.py',
                                'action' : rp.TRANSFER
                               }
                              ]
        cud1.output_staging = [{'source' : 'unit:///classified_images',
                                'target' : 'pilot:///classified_images',
                                'action' : rp.LINK}
                              ]
        
        
        units = umgr.submit_units(cud1)

        # Wait for all compute units to reach a final state (DONE, CANCELED or FAILED).
        report.header('gather results')
        umgr.wait_units()

        cud2 = rp.ComputeUnitDescription()
        cud2.executable    = 'python3'
        cud2.gpu_processes = 1
        cud2.cpu_processes = 1
        cud2.pre_exec = ['module load psc_path/1.1',
                         'module load slurm/default',
                         'module load intel/17.4',
                         'module load python3',
                         'module load cuda',
                         'module load opencv',
                         'source /pylon5/mc3bggp/paraskev/pytorchCuda/bin/activate',
                         'mkdir -p blob_detected']

        cud2.executable = ['python3']   # Assign executable to the task   
        cud2.arguments = ['blob_detector.py']
        cud2.input_staging = [{'source' : 'pilot:///classified_images',
                               'target' : 'unit:///classified_images',
                               'action' : rp.LINK},
                               {'source' : 'client:///blob_detector.py',
                                'target' : 'unit:///blob_detector.py',
                                'action' : rp.TRANSFER}
                              ]
        
        units = umgr.submit_units(cud2)

        # Wait for all compute units to reach a final state (DONE, CANCELED or FAILED).
        report.header('gather results')
        umgr.wait_units()
        
    except Exception as e:
        # Something unexpected happened in the pilot code above
        report.error('caught Exception: %s\n' % e)
        ru.print_exception_trace()
        raise

    except (KeyboardInterrupt, SystemExit) as e:
        # the callback called sys.exit(), and we can here catch the
        # corresponding KeyboardInterrupt exception for shutdown.  We also catch
        # SystemExit (which gets raised if the main threads exits for some other
        # reason).
        ru.print_exception_trace()
        report.warn('exit requested\n')

    finally:
        # always clean up the session, no matter if we caught an exception or
        # not.  This will kill all remaining pilots.
        report.header('finalize')
        session.close(download=True)

    report.header()
