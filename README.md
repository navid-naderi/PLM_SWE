# Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport

This repository contains the implementation code for our preprint [Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport](https://www.biorxiv.org/content/10.1101/2024.01.29.577794v1.abstract), which showcases the benefits of sliced-Wasserstein embedding to summarize token-level representations produced by pre-trained ESM-2 protein language models (PLMs).

## Abstract

Protein language models (PLMs) have emerged as powerful approaches for mapping protein sequences into informative embeddings suitable for a range of applications. PLMs, as well as many other protein representation schemes, generate per-token (i.e., per-residue) representations, leading to variable-sized outputs based on protein length. This variability presents a challenge for protein-level prediction tasks, which require uniform-sized embeddings for consistent analysis across different proteins. Prior work has typically resorted to average pooling to summarize token-level PLM outputs. It is, however, unclear if such an aggregation operation effectively prioritizes the relevant information across token-level representations. Addressing this, we introduce a novel method utilizing sliced-Wasserstein embeddings to convert variable-length PLM outputs into fixed-length protein-level representations. Inspired by the success of optimal transport techniques in representation learning, we first conceptualize per-token PLM outputs as samples from a probabilistic distribution. We then employ sliced-Wasserstein distances to map these samples against a learnable reference set, creating a Euclidean embedding in the output space. The resulting embedding is agnostic to the length of the input and represents the entire protein. Across a range of state-of-the-art pre-trained ESM-2 PLMs, with varying model sizes, we show the superiority of our method over average pooling for protein-drug and protein-protein interaction. Our aggregation scheme is especially effective when model size is constrained, enabling smaller-scale PLMs to match or exceed the performance of average-pooled larger-scale PLMs. Since using smaller models reduces computational resource requirements, our approach not only promises more accurate inference but can also help democratize access to foundation models.

## Setting Up the Environment

Create a new virtual Conda environment, called `plm_swe`, with the required libraries using the following commands:

```
conda create -n plm_swe python=3.9
conda activate plm_swe
pip install -r requirements.txt
```

## Downloading the Datasets
The `download_data.py` script can be used to download the datasets for the experiments into a new folder called `datasets` by running

```
python download_data.py —-to datasets —-benchmarks davis bindingdb ppi_gold
```

## Running the Numerical Experiments
The following commands can be used to run the four numerical experiments in the paper. The number of points in the reference set, number of slices, and the size of the ESM-2 pre-trained PLM can be adjusted via command-line parameters, as well as the configuration files under `config`.

### Drug-Target Interaction (DAVIS)
```
python run_dti.py --run-id dti_davis_swepooling_128refpoints_128slices_esm2_8m --config config/dti_davis.yaml --num-ref-points 128 --num-slices 128 --target-model-type esm2_t6_8M_UR50D
```

### Drug-Target Interaction (Binding-DB)
```
python run_dti.py --run-id dti_bindingdb_swepooling_128refpoints_128slices_esm2_8m --config config/dti_bindingdb.yaml --num-ref-points 128 --num-slices 128 --target-model-type esm2_t6_8M_UR50D
```

### Out-of-Domain Drug-Target Affinity
```
python run_dti.py --run-id dti_tdc_dg_swepooling_128refpoints_128slices_esm2_8m --config config/dti_tdc_dg.yaml --num-ref-points 128 --num-slices 128 --target-model-type esm2_t6_8M_UR50D
```

### Protein-Protein Interaction
```
python run_ppi.py --run-id ppi_gold_swepooling_128refpoints_128slices_esm2_8m --config config/ppi_gold.yaml --num-ref-points 128 --num-slices 128 --target-model-type esm2_t6_8M_UR50D
```

# Acknowledgments

This repository is based on the following repositories:
- [ConPLex](https://github.com/samsledje/ConPLex)
- [PSWE](https://github.com/navid-naderi/PSWE)
- [torchinterp1d](https://github.com/aliutkus/torchinterp1d)
- [SLOSH](https://github.com/mint-vu/SLOSH)

# Citation

If you make use of this repository, please cite our preprint using the following BibTeX format:
```
@article{naderializadeh2024_plm_swe,
  title={Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport},
  author={NaderiAlizadeh, Navid and Singh, Rohit},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
