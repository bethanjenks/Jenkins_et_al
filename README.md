# Jenkins_et_al

Code accompanying:

**Jenkins et al.**  
*Distributed representations of chemosensory valence in a naïve vertebrate brain*

## Overview
This repository contains analysis code and notebooks used for whole-brain neural imaging and behavioural analysis in larval zebrafish.

The repository includes analysis pipelines for:

- neural stimulus and behaviour regression analyses
- valence neuron identification and characterization
- functional clustering and spatial organization of neurons
- whole-brain population activity analysis and visualization
- stimulus decoding and classification
- functional connectivity analyses
- behavioural analysis and quantification

## Repository Structure

```text
notebooks/          # exploratory analyses, figure generation (frozen at commit 2d19f3a)
scripts/            # reusable analysis pipelines and helper scripts (frozen at commit 2d19f3a)
src/jenkins_et_al/  # in-progress package porting the above into loading/processing/plotting
                    # functions, verified against golden captures of the originals.
                    # See PORTING_PLAN.md for status.
```

## Main Analyses

### Neural analyses

- Stimulus-regressor correlation analysis
- Behaviour-regressor correlation analysis
- Functional clustering analyses
- Whole-brain population analysis
- Stimulus decoding and classification analyses
- Noise correlation analysis 
- Spatial segregation analysis (LDA/SVM)
  

### Behaviour analyses

- Stimulus preference analysis
- Bout frequency analysis
- Tail vigour analysis
- Turning/laterality analysis


## Requirements

Main Python packages used:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

## Data

Raw imaging and behavioural datasets are not included in this repository.

## Citation

If using this code, please cite:
Jenkins et al. (in preparation)
