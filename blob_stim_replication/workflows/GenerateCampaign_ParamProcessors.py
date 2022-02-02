# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        C. Pokorny
# Date:          02/2022

import json
import os
import shutil


def stim_file_from_template(*, path, stim_file_template, **kwargs):
    """Places a stimulation spike file based on an existing template into
       the simulation folders:
       The template can either be
         - a spike file (.dat)
         - another BlueConfig (.json) from which the path of the spike file
           is loaded; a copy of that BlueConfig will be added to <path>, as
           it may contain additional information about that spike file.
    """

    # Define target spike file
    stim_filename = 'input.dat'
    stim_file = os.path.join(path, stim_filename)

    # Define source spike file
    if os.path.splitext(stim_file_template)[-1].lower() == '.json':
        with open(stim_file_template, 'r') as f:
            stim_dict = json.load(f)
        src_file = stim_dict[0]['stim_file'] # Lookup stimulus file from config file

        cfg_file = os.path.join(path, '_STIM_FILE_TEMPLATE_' + os.path.split(stim_file_template)[-1])
        shutil.copyfile(stim_file_template, cfg_file) # Make copy of that config file
    else:
        src_file = stim_file_template # Template file is stimulus spike file

    # Copy source to target spike file
    shutil.copyfile(src_file, stim_file)

    return {'stim_file': stim_filename}
