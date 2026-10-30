# Data and model provenance

The package includes three gzip-compressed feature tables and two portable fitted classifiers under [`src/rot_features/assets`](../src/rot_features/assets). It contains no dataset photographs or neural-network checkpoints. CSV stores floating-point values with round-trip precision; JSON describes coefficients, scaling, selected features, and tree structures without executable pickle content.

| Table | Complete rows | Positive | Negative | Use |
|---|---:|---:|---:|---|
| RoT-DS | 3,638 | 1,836 | 1,802 | Fit and test the RoT-DS classifier. |
| KU-PCP | 3,516 | 788 | 2,728 | Fit and test the KU-PCP classifier. |
| AVT | 1,115 | 204 | 911 | External evaluation of the RoT-DS classifier. |

These are the usable rows in the author’s original cleaned tables, which can be fewer than the image counts described in the dataset papers. `source_row` preserves row identity and order from the archived table; it is not a filename or a new random index. Each model records the exact source rows in its test split.

The original source-table SHA-256 is `8b4c1f35075bd4647148a30e98f626fa6d8f7d0d9e590b73501bf48364f63f8f`; the fitted-state archive SHA-256 is `02fadaab65c91270f4dc0cd846dc7b9359a08cb0c1f4d6c005fff9af6f7c7c43`. [manifest.json](manifest.json) records every exported file hash. The authoritative copy is packaged with the code; `rot-reproduce verify` checks all five artifacts before evaluation.

The source datasets are RoT-DS from Mai et al., using the subset archived by Göring and Raake; KU-PCP from Lee et al.; and the AVT image database from Göring and Raake. AVT labels in these tables use the majority of five raters. Their sources and experimental use are described in [the paper](https://doi.org/10.1145/3746273.3760202). Obtain images from their original providers and observe each dataset’s terms. The exported features do not grant rights to the photographs.

- RoT-DS and AVT: Göring and Raake’s [data archive](https://zenodo.org/records/5140038) contains `rule_of_thirds.tar.gz` and `avt_image_db.tar.gz`. Their [repository](https://github.com/Telecommunication-Telemedia-Assessment/sophoappeal_rule_prediction) provides the corresponding annotations and source context.
- KU-PCP: Lee et al.’s [Semantic Lines and Photographic Composition project](https://mcl.korea.ac.kr/jtlee_jvcir2018/) is the original dataset source.

[provenance.json](provenance.json) identifies the recovered feature implementation and evaluation notebook. The original private experiment folders and Python environments are not required to replay the bundled artifacts.

[evaluation_images.csv](evaluation_images.csv) maps the 2,547 recorded evaluation rows to dataset-relative image paths and SHA-256 hashes. It preserves the extraction order used by the two validation processes; dividing alternate rows reconstructs their two manifests. [Image setup](../docs/image_pipeline.md#recompute-the-recorded-evaluation-images) explains how to verify the images and reproduce feature extraction without machine-specific paths.
