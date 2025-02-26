> [!WARNING]
> The Blue Brain Project concluded in December 2024, so development has ceased under the BlueBrain GitHub organization.
> Future development will take place at: https://github.com/openbraininstitute/o1v5-workflows

# O1v5-workflows

Workflow configurations and extensions for setting up and running simulation campaigns using the O1v5 circuit

## Folder organization:
* __[/blob_stim_replication](/blob_stim_replication)__\
  Code and config files for setting up and running replication experiments of the BlobStim experiment (proj32) using the converted O1v5-SONATA circuit [NSETM-1222] with original TC (BlobStim) projections under various connectome manipulations.

* __[/simplified_connectome_models](/simplified_connectome_models)__\
  Code and config files for setting up and running replication experiments of the BlobStim experiment (proj32) using the converted O1v5-SONATA circuit [NSETM-1222] with original TC (BlobStim) and simplified connectomes.

* __[/project_cleanup](/project_cleanup)__\
  Code and workflows for project cleanup, i.e., moving simulation campaigns to other folder locations, etc.

## Requirements:
* [bbp-workflow CLI](https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?spaceKey=BBPNSE&title=Workflow)

## How to set up and run simulation campaigns:
**1. Register a new circuit** (only once, if not yet registered in Nexus) <br>
<code>bbp-workflow launch --follow --config workflows/RegisterCircuit__xxx.cfg bbp_workflow.circuit.task RegisterDetailedCircuit</code>

  To be specified in <code>RegisterCircuit__xxx.cfg</code>:
  * Circuit name, type, description, and additional information
  * Circuit config

**2. Set up simulation campaign** <br>
<code>bbp-workflow launch --follow --config workflows/GenerateCampaign__xxx.cfg bbp_workflow.simulation GenerateSimulationCampaign</code>

  To be specified in <code>GenerateCampaign__xxx.cfg</code>:
  * Campaign name and description
  * Circuit URL
  * Coordinates (parameters that are varied throughout the campaign)
  * Attributes (fixed parameters, including campaign path and BlueConfig template)
  * Parameter processors (e.g., user target generation, stimulus generation, ...)

**3. Generate (human-readable) symbolic links** <br>
<code>bbp-workflow launch --follow --config workflows/GenerateSymLinks__xxx.cfg bbp_workflow.simulation SimCampaignInfo generate-symlinks</code>

  To be specified in <code>GenerateSymLinks__xxx.cfg</code>:
  * Campaign URL

**4. Launch simulation campaign** <br>
<code>bbp-workflow launch --follow --config workflows/LaunchCampaign__xxx.cfg bbp_workflow.simulation SimulationCampaign</code>

  To be specified in <code>LaunchCampaign__xxx.cfg</code>:
  * Campaign URL
  * Project account for SLURM allocation
  * Simulation type
  * Hardware resources and allocation time

ℹ️ Note: Adding <code>benchmark=True</code> at the end of the launch command will only run the last simulation in a campaign, which can be useful for benchmarking.

**5. Analyze/visualize simulation campaigns** <br>
[Analysis launcher workflow](https://bbpgitlab.epfl.ch/conn/simulation/sscx-analysis/-/tree/master/analysis_launcher) on GitLab

## IMPORTANT:
* Unless <code>--workflows-dir</code> is specified, <code>bbp-workflow ...</code> must be launched from the root folder containing <code>./workflows</code> as a subfolder!
* The Nexus instance (staging or production) can be selected in the <code>[DEFAULT]</code> section in the config files!

# Funding & Acknowledgment
This development is supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.
Copyright © 2024 Blue Brain Project/EPFL
