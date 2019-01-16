from seals.entk_script.entk_script import run
import radical.utils as ru
import pandas as pd

if __name__ == '__main__':

    run(resource='xsede.bridges', 
        walltime=60, 
        cpus=32, 
        gpus=2, 
        project=, 
        queue='GPU', 
        name=ru.generate_id('seals.validation', ru.ID_PRIVATE), 
        input_dir='/pylon5/mc3bggp/bspitz/Validation/',
        scale_bands=299, 
        model)
    
    images = pd.read_csv('images.csv')

    for image in images:
        # Check the output with the expected one. Report error otherwise
        print image
