# Acknowledgments and upstream provenance

RHC-Drive extends the [R2-Dreamer implementation](https://github.com/NM512/r2dreamer) with response-history context learning, auxiliary response prediction and context exchange, posterior conditioning, vehicle randomization, and driving experiment workflows. The original MIT copyright notice is preserved in [LICENSE](LICENSE). Existing algorithm names and configuration keys retain upstream identifiers where required for compatibility.

If your work uses the R2-Dreamer backbone, please also credit its authors:

```bibtex
@inproceedings{morihira2026rdreamer,
  title = {R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
  author = {Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year = {2026},
  url = {https://openreview.net/forum?id=Je2QqXrcQq}
}
```

Driving environments and vehicle assets come from [MetaDrive](https://github.com/metadriverse/metadrive). nuPlan-derived scenarios require separately prepared data and their applicable source licenses. Simulator source and asset provenance are tracked by the runtime export manifest; those assets are not bundled in this Git repository.

The paper-method figures in [assets/figures](assets/figures/) are drawn from the RHC-Drive manuscript. Their source filenames and rendering details are recorded in [assets/README.md](assets/README.md).
