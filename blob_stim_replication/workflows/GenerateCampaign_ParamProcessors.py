# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        C. Pokorny
# Date:          02/2022
# Last modified: 12/2022

import json
import hashlib
import numpy as np
import pandas as pd
import os
import shutil
from bluepy import Circuit
import lookup_projection_locations as projloc
import stimulus_generation as stgen


""" Generates user target file, combining targets from custom list and projection paths """
def generate_user_target(*, circuit_config, path, custom_user_targets=[], **kwargs):

    circuit = Circuit(circuit_config)
    proj_paths = list(circuit.config['projections'].values())
    proj_targets = [os.path.join(os.path.split(p)[0], 'user.target') for p in proj_paths]
    proj_targets = list(filter(os.path.exists, proj_targets))
    
    target_paths = custom_user_targets + proj_targets
    
    user_target_name = 'user.target'
    user_target_file = os.path.join(path, user_target_name)
    with open(user_target_file, 'w') as f_tgt:
        for p in target_paths:
            assert os.path.exists(p), f'ERROR: Target "{p}" not found!'
            with open(p, 'r') as f_src:
                f_tgt.write(f_src.read())
                f_tgt.write('\n\n')
            # print(f'INFO: Adding target "{p}" to "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    # Set group membership to same as <path> (should be 10067/"bbp")
    # os.chown(user_target_file, uid=-1, gid=os.stat(path).st_gid) # [NOT NEEDED ANY MORE?]
    
    # print(f'INFO: Generated user target "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    return {'user_target_name': user_target_name}


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


def get_cfg_hash(cfg):
    """
    Generates MD5 hash code for given config dict
    """
    hash_obj = hashlib.md5()

    def sort_dict(data):
        """
        Sort dict entries recursively, so that the hash function
        does not depend on the order of elements
        """
        if isinstance(data, dict):
            return {key: sort_dict(data[key]) for key in sorted(data.keys())}
        else:
            return data

    # Generate hash code from sequence of cfg params (keys & values)
    for k, v in sort_dict(cfg).items():
        hash_obj.update(str(k).encode('UTF-8'))
        hash_obj.update(str(v).encode('UTF-8'))
    
    return hash_obj.hexdigest()


def generate_random_dot_flash_stimulus_v5(*, path, sim_duration, **kwargs):
    """
    Generates 'random dot flash' type of stimulus file and writing the
    spikes file(s) to hashed folders to prevent multiple generation of
    exact same spike files
    => Backward compatible v5 version, accepting a .json file as projection name
    """

    # _Init_

    ## Get stim config parameters and hash code
    param_list = ['circuit_config', # Circuit
                  'proj_name', 'proj_mask', 'proj_mask_type', 'proj_flatmap', 'num_fibers_per_cluster', 'stimuli_seeds', 'sparsity', # Spatial params
                  'num_stimuli', 'series_seed', 'strict_enforce_p', 'p_seeds', 'overexpressed_tuples', # Series params
                  'start', 'duration_stim', 'duration_blank', 'rate_min', 'rate_max', # Rate signal params
                  'spike_seed', 'bq', 'tau'] # Spike params

    cfg = {'stim_name': 'RandomDotFlash'}
    cfg.update({p: kwargs.get(p) for p in param_list}) # Stim config parameters
    cfg_hash = get_cfg_hash(cfg)

    ## Define paths and files
    spikes_path = os.path.join(os.path.split(path)[0], 'spikes', cfg_hash)
    figs_path = os.path.join(spikes_path, 'figs')
    stim_file = os.path.join(spikes_path, 'input.dat')
    rel_stim_file = os.path.relpath(stim_file, path) # Relative path to current simulation folder
    props_file = os.path.splitext(stim_file)[0] + '.json'

    ## Check if stimulus folder for given parameter configuration already exists (using hash code as folder name)
    ## [IMPORTANT: Thread-safe implementation w/o os.path.exists followed by os.makedirs, since this would create
    ##             a race condition in case max_workers > 1 parallel processes are used!]
    try:
        os.makedirs(spikes_path)
        # os.chown(spikes_path, uid=-1, gid=os.stat(path).st_gid) # Set group membership same as <path> (should be 10067/"bbp") # [NOT NEEDED ANY MORE?]

        os.makedirs(figs_path)
        # os.chown(figs_path, uid=-1, gid=os.stat(path).st_gid) # Set group membership same as <path> (should be 10067/"bbp") # [NOT NEEDED ANY MORE?]

    except FileExistsError: # Stimulus folder already generated, stop here!
        print(f'INFO: Stim folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]} already exists ... SKIPPING!')
        return {'stim_file': rel_stim_file, 'stim_name': cfg['stim_name']}
    
    print(f'INFO: Generating "{cfg["stim_name"]}" stimulus in folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]}!')

    ## Load circuit
    circ = Circuit(cfg['circuit_config'])

    user_target_name = kwargs.get('user_target_name', '')
    user_target_file = os.path.join(path, user_target_name)
    if len(user_target_name) > 0 and os.path.exists(user_target_file):
        # Make individual user targets generated by generate_user_target available
        circ_cfg_dict = circ.config.copy()
        circ_cfg_dict['targets'].append(user_target_file) # Add user target file to list of existing target files
        circ = Circuit(circ_cfg_dict) # Re-load circuit

    # print(f'INFO: Loaded circuit with {len(circ.cells.targets)} targets!')

    # _Step1_: Define stimuli (spatial structure)

    ## Get fiber GIDs and locations
    if os.path.isfile(cfg['proj_name']) and os.path.splitext(cfg['proj_name'])[-1] == '.json':
        # Load fibers from .json file [BACKWARD COMPATIBILITY with O1v5 BlobStim fibers]
        # Unused config properties: proj_mask, proj_mask_type, proj_flatmap, num_fibers_per_cluster
        assert cfg['proj_mask'] is None  and cfg['proj_mask_type'] is None and cfg['proj_flatmap'] is None and cfg['num_fibers_per_cluster'] is None, 'ERROR: Unused properties mut not be set!' # Strict assertion, just to avoid any confusion
        with open(cfg['proj_name'], 'r') as f:
            proj_dict = json.load(f)

        grp_gids = [np.array(proj_dict['groups'][f'group{g}']['gids']) for g in range(len(proj_dict['groups']))]
        grp_pos = np.array([np.array(proj_dict['groups'][f'group{g}']['location']) for g in range(len(proj_dict['groups']))])
        gids = np.unique(np.concatenate(grp_gids))
        grp_idx = np.full_like(gids, -1)
        for gidx, gg in enumerate(grp_gids):
            gsel = np.isin(gids, gg)
            assert np.all(grp_idx[gsel] == -1), 'ERROR: Multiple group assignments per fiber!'
            grp_idx[gsel] = gidx

        pos2d = None
        # No single-fiber locations/directions available => Replicate group locations/directions instead (mainly used for visualizations)
        pos3d = np.array([proj_dict['groups'][f'group{g}']['location'] for g in grp_idx])
        dir3d = np.array([proj_dict['groups'][f'group{g}']['direction'] for g in grp_idx])

        # There is no distinction between selected and all fibers (no mask)
        pos2d_all = pos2d
        pos3d_all = pos3d
    else:
        gids, pos2d, pos3d, dir3d = projloc.get_projection_locations(circ, cfg['proj_name'], cfg['proj_mask'], cfg['proj_mask_type'], cfg['proj_flatmap'])

        ## Cluster groups of nearby fibers (blobs) based on 3D locations [DON'T USE 2D LOCATIONS, since they may be discretized and contain duplicates due to flatmap conversion]
        grp_gids, grp_pos, grp_idx = projloc.cluster_by_locations(gids, pos3d, n_per_cluster=cfg['num_fibers_per_cluster'])
        
        ## Positions of all fibers w/o mask (for plotting only)
        _, pos2d_all, pos3d_all, _ = projloc.get_projection_locations(circ, cfg['proj_name'], None, None, cfg['proj_flatmap'])

    ## Plot groups of fibers
    projloc.plot_clusters_of_fibers(grp_idx, grp_pos, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

    ## Plot cluster size distribution
    projloc.plot_cluster_size_distribution(grp_idx, figs_path)

    ## Generate spatial patterns
    pattern_grps, pattern_gids, pattern_pos2d, pattern_pos3d = stgen.generate_spatial_pattern(gids, pos2d, pos3d, grp_idx, cfg['stimuli_seeds'], cfg['sparsity'])

    ## Plot spatial patterns
    stgen.plot_spatial_patterns(pattern_pos2d, pattern_pos3d, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

    # _Step2_: Define stim series (stim train)

    ## Generate stimulus series (stim train)
    num_patterns = len(pattern_grps)
    stim_train = stgen.generate_stim_series(num_patterns, cfg['num_stimuli'], cfg['series_seed'], cfg['strict_enforce_p'], cfg['p_seeds'], cfg['overexpressed_tuples'])

    ## Plot stim train
    stgen.plot_stim_series(stim_train, figs_path)

    # _Step3_: Define analog rate signal (based on outputs of steps 1 & 2)

    ## Generate analog rate signals map (per group of fibers)
    num_groups = len(grp_gids)
    rate_map, time_axis, time_windows = stgen.generate_rate_signals(stim_train, pattern_grps, num_groups, cfg['start'], cfg['duration_stim'], cfg['duration_blank'], cfg['rate_min'], cfg['rate_max'])

    if time_windows[-1] > sim_duration:
        print('WARNING: Generated stimulus signals longer than simulation duration!')

    ## Plot analog rate signals
    stgen.plot_rate_signals(rate_map, time_axis, stim_train, time_windows, figs_path)

    # _Step4_: Define spike trains (temporal structure; based on output of step 3)

    ## Generate spikes for each group
    spike_map = stgen.generate_spikes(rate_map, time_axis, cfg['spike_seed'], cfg['bq'], cfg['tau'])

    ## Plot spikes for each group
    stgen.plot_spikes(spike_map, time_axis, stim_train, time_windows, 'Group idx', False, figs_path)

    ## Map groups to fibers
    out_map = stgen.map_groups_to_fibers(spike_map, grp_gids)

    ## Plot output spikes per fiber & stimulus PSTHs
    stgen.plot_spikes(out_map, time_axis, stim_train, time_windows, 'Fiber GIDs', False, figs_path)
    stgen.plot_PSTHs(out_map, stim_train, time_windows, 10, figs_path)

    # _Step5_: Write spike output & properties files (based on output of step 4)
    
    ## Spike file, containing actual spike trains
    stgen.write_spike_file(out_map, stim_file)
    
    ## Properties file, containing config params and properties of generated stimulus
    props = {'grp_gids': grp_gids, 'grp_pos': grp_pos, 'pattern_grps': pattern_grps,
             'stim_train': stim_train, 'time_windows': time_windows}

    def np_conv(data):
        """ Convert numpy.ndarray to list recursively, so that JSON serializable """
        if isinstance(data, dict):
            return {k: np_conv(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [np_conv(d) for d in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    with open(props_file, 'w') as f:
        json.dump({'cfg': cfg, 'props': np_conv(props)}, f, indent=2)

    # Can be easily derived from these properties:
    # pattern_gids = [np.hstack([grp_gids[grp] for grp in grps]) for grps in pattern_grps]

    return {'stim_file': rel_stim_file, 'stim_name': cfg['stim_name']}


def config_section_from_dict(sect_type, sect_name, param_dict, intend=4):
    """
    Generates a BlueConfig section string from a dict
    """
    section_str = sect_type + ' ' + sect_name + '\n{\n'
    for k, v in param_dict.items():
        section_str += ' ' * intend + f'{k} {str(v)}\n' 
    section_str += '}\n\n'
    return section_str


def remove_connections(*, path, seed, circuit_config, circuit_target, remove_conns_list, remove_conns_mode, remove_conns_amount, remove_conns_seed, remove_conns_seed_mode, custom_user_targets=[], **kwargs):
    """ Param-processor to remove connections between neurons, by creating "Connections" blocks with weights set to zero
          remove_conns_list: List of exactly one pre and one post cell target name, or
                             list of connections containing lists of pre/post cell GIDs, or
                             .csv file (2 columns with src/tgt GIDs, no header) with list of connections, or
                             .npy file (<2xN> array with src/tgt GIDs) with list of connections
          remove_conns_mode: "directed": List of directed connections assumed; a certain amount of connections if removed
                             "reciprocal": List of reciprocal connections assumed; a certain amount of reciprocal connections is replaced by single edges choosing their direction at random
          remove_conns_amount: Amount of connections to be removed, either defined as fraction (float between 0.0 and 1.0) or absolute number (integer >= 0)
                               Can be provided as single value, or list of values in which case campaign generator will cycle throu this list taking different numbers for different simulations
          remove_conns_seed: Seed for randomly selecting connections to be removed
          remove_conns_seed_mode: "constant" ... Use same seed for all simulations
                                  "add_sim_idx" ... Add sim index for seeding each simulation, i.e., different removal seeds across simulations
                                  "add_sim_seed" ... Add sim seed for seeding each simulation, i.e., different removal seeds across simulations, depending on simulation seed
          Returns: String with "Connection" blocks for the BlueConfig
                   Target file name added to custom_user_targets [generate_user_target param-processor must be run AFTERWARDS]
          
          NOTE: For the same remove_conns_seed, the exact same (random) selection of connections (choice & direction) will consistently be included in higher amounts (plus some new ones)!
    """
    sim_idx = int(os.path.split(path)[-1])

    if isinstance(remove_conns_list, list):
        if np.ndim(remove_conns_list) == 1: # OPTION 1: List of pre/post target specifications
            assert len(remove_conns_list) == 2, 'ERROR: Pre/post cell target specification pair required!'
            c = Circuit(circuit_config)
            pre = np.intersect1d(c.cells.ids(circuit_target), c.cells.ids(remove_conns_list[0]))
            post = np.intersect1d(c.cells.ids(circuit_target), c.cells.ids(remove_conns_list[1]))
            conns_all = pd.DataFrame(list(c.connectome.iter_connections(pre=pre, post=post)))
        else: # OPTION 2: List of connections (GID pairs)
            assert np.ndim(remove_conns_list) == 2, 'ERROR: 2D list of pre/post GID pairs required!'
            conns_all = pd.DataFrame(remove_conns_list)
    elif isinstance(remove_conns_list, str): # OPTION 3: Load connections (GID pairs) from file
        if os.path.splitext(remove_conns_list)[-1].lower() == '.csv': # (3a) .csv file with 2 columns
            conns_all = pd.read_csv(remove_conns_list, header=None)
        elif os.path.splitext(remove_conns_list)[-1].lower() == '.npy': # (3b) .npy file with <2xN> array
            conns_all = pd.DataFrame(np.load(remove_conns_list).T)
        else:
            assert False, 'ERROR: Only ".csv" or ".npy" files supported!'
    else:
        assert False, 'ERROR: remove_conns_list must be a list or filename!'

    assert conns_all.shape[1] == 2, 'ERROR: remove_conns_list must contain two columns (pre/post)!'

    # Determine amount of connections to be removed
    if isinstance(remove_conns_amount, list):
        if sim_idx >= len(remove_conns_amount):
            print(f'WARNING: Simulation count exceeds length of amount list - REPEATING FROM BEGINNING!')
        amount = remove_conns_amount[np.mod(sim_idx, len(remove_conns_amount))]
    else:
        amount = remove_conns_amount
    N_all = conns_all.shape[0]
    if isinstance(amount, float):
        assert 0.0 <= amount <= 1.0, 'ERROR: Amount (fraction) out of range!'
        N_sel = np.round(N_all * amount).astype(int)
    elif isinstance(amount, int):
        assert amount >= 0, 'ERROR: Amount (count) out of range!'
        N_sel = np.minimum(N_all, amount)
        if N_sel < amount:
            print('WARNING: Not enouch connections to be removed!')
    else:
        assert False, 'ERROR: Amount(s) must be of type "float" or "int"!'

    # Select connections to be removed
    if remove_conns_seed_mode == 'constant':
        rmv_seed = remove_conns_seed
    elif remove_conns_seed_mode == 'add_sim_idx':
        rmv_seed = remove_conns_seed + sim_idx
    elif remove_conns_seed_mode == 'add_sim_seed':
        rmv_seed = remove_conns_seed + seed
    else:
        assert False, f'ERROR: Seed mode "{remove_conns_seed_mode}" unknown!'
    np.random.seed(rmv_seed)
    sel_idx = np.random.choice(N_all, N_sel, replace=False)
    sort_idx = np.argsort(sel_idx)
    sel_idx = sel_idx[sort_idx] # Sort selected connections by index
    conns_sel = conns_all.iloc[sel_idx]

    # Select connection direction to be removed
    if remove_conns_mode == 'directed':
        dir_sel = np.zeros_like(sel_idx) # Remove connection as it is
    elif remove_conns_mode == 'reciprocal':
        dir_sel = np.random.choice(2, len(sel_idx)) # Select one direction at random to be removed
    else:
        assert False, f'ERROR: remove_conns_mode "{remove_conns_mode}" unknown!'
    dir_sel = dir_sel[sort_idx] # Sort direction selection as well, so that consistent across different numbers of connections to be removed (with same seed)
    print(f'INFO: <SIM{sim_idx}> Randomly selected {N_sel} of {N_all} {remove_conns_mode} connections ({np.sum(dir_sel)} directions flipped) to be removed (amount={amount}, seed={rmv_seed})!')

    # Remove connections, creating (i) "Connection" blocks and (ii) cell targets
    conns_blocks = ''
    conns_removed = []
    target_file = os.path.join(path, 'conns_removed.target')
    with open(target_file, 'a') as fid:
        for idx, c in enumerate(conns_sel.to_numpy()):
            if dir_sel[idx] == 1:
                c = np.flip(c) # Flip direction

            # Create "Connection" block with weight zero
            target_name = f'ConnRemoved_{idx}'
            param_dict = {'Source': target_name + '_src', 'Destination': target_name + '_dest', 'Weight': 0.0, 'SpontMinis': 0.0}
            conns_blocks += config_section_from_dict('Connection', target_name, param_dict, intend=4)
            conns_removed.append(c)

            # Create named single-cell src/dest targets
            target_str = f'Target Cell {target_name}_src\n{{\na{c[0]}\n}}\n\n'
            target_str += f'Target Cell {target_name}_dest\n{{\na{c[1]}\n}}\n\n'

            # Write to .target file
            fid.write(target_str)

    # Write to .csv & .npy files
    pd.DataFrame(conns_removed).to_csv(os.path.join(path, 'conns_removed.csv'), index=False, header=False)
    np.save(os.path.join(path, 'conns_removed.npy'), np.array(conns_removed).T)

    # Remove "conns_removed.target" user targets files, if existing from other simulations
    custom_user_targets = list(filter(lambda t: os.path.split(target_file)[1] != os.path.split(t)[1], custom_user_targets))

    # Add .target file to user targets
    if N_sel > 0:
        custom_user_targets.append(target_file)

    return {'removed_conns_blocks': conns_blocks, 'custom_user_targets': custom_user_targets}


def remove_outgoing_connections(*, path, circuit_config, circuit_target, remove_conns_src_list, remove_conns_block_table=None, custom_user_targets=[], **kwargs):
    """ Param-processor to remove outgoing connections from a list of source neurons (optionally, based on some block design)
          remove_conns_src_list: List of source GIDs whose outgoing connections to be removed
          remove_conns_block_table (OPTIONAL): .npy file containing boolean <#src x #sims> array with boolean selections of source neurons (from remove_conns_src_list) in each simulation
          Returns: String with "conns_remove_src" target name
                   Target file containing that target added to custom_user_targets [generate_user_target param-processor must be run AFTERWARDS]
    """
    sim_idx = int(os.path.split(path)[-1])

    if remove_conns_block_table is None:
        sel_idx = np.ones(len(remove_conns_src_list), dtype=bool) # Select all
    else:
        block_table = np.load(remove_conns_block_table)
        assert block_table.shape[0] == len(remove_conns_src_list), 'ERROR: Block design does not match source GID list!'
        if sim_idx >= block_table.shape[1]:
            print(f'WARNING: Simulation count exceeds length of block design - REPEATING FROM BEGINNING!')
        sel_idx = block_table[:, np.mod(sim_idx, block_table.shape[1])]
    gids_sel = np.array(remove_conns_src_list)[sel_idx]
    bin_code = int(''.join(sel_idx.astype(int).astype(str)), 2) # Convert binary pattern to integer [just for logging/testing]
    print(f'INFO: <SIM{sim_idx}> Selected {len(gids_sel)} of {len(sel_idx)} sources of outgoing connections to be removed (selection code {bin_code})!')

    # Create cell target
    hash_obj = hashlib.shake_128()
    hash_obj.update(str(gids_sel).encode('UTF-8'))
    hash_code = hash_obj.hexdigest(4).upper()

    target_file = os.path.join(path, 'outgoing_conns_removed.target')
    cell_target_name = 'OutgoingConnsRemovedSrc_' + hash_code
    with open(target_file, 'a') as fid:
        gids_str = ' '.join([f'a{gid}' for gid in gids_sel])
        target_str = f'Target Cell {cell_target_name}\n{{\n{gids_str}\n}}\n'
        fid.write(target_str)
    print(f'Cell target "{cell_target_name}" with {len(gids_sel)} cells written to {target_file}')

    # Write to .csv & .npy files
    pd.DataFrame(gids_sel).to_csv(os.path.join(path, 'outgoing_conns_removed.csv'), index=False, header=False)
    np.save(os.path.join(path, 'outgoing_conns_removed.npy'), gids_sel)

    # Remove "outgoing_conns_removed.target" user targets files, if existing from other simulations
    custom_user_targets = list(filter(lambda t: os.path.split(target_file)[1] != os.path.split(t)[1], custom_user_targets))

    # Add .target file to user targets
    if len(gids_sel) > 0:
        custom_user_targets.append(target_file)

    return {'conns_remove_src': cell_target_name, 'custom_user_targets': custom_user_targets}