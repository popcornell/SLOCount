# SLOCount
Deep Learning driven Speaker Counting exploiting multi-channel Source LOCalization (SLOC) features.

---
The neural architecture is defined in src/architecture.py. 

---

The full synthetic data development set, created with LibriSpeech dev-clean is available at: 
[Onedrive_link](https://univpm-my.sharepoint.com/:u:/g/personal/s1080434_studenti_univpm_it/EdZKJH_2kItBnnACJo4o-lwBqWK3xuLAblPKXzGwiTJa8A?e=YuorBJ).

This version has a mean SIR of 2.8 dB. The total size is 8.20 GB. 

The train and eval synthetic datasets are available on request. 

--- 
To run: 

- download the synthetic dataset.
- adjust the paths in configs/local_chime.yaml.
- run trainer.py

--- 

Alternatively we provide the alignments to produce : 
- Download the CHiME-5 data.
- Download our tri-3 Kaldi alignments for the CHiME-5 data at [Onedrive-link](https://univpm-my.sharepoint.com/:u:/g/personal/s1080434_studenti_univpm_it/EX-YfhzYLCVKq1zRgc5hJwMBTWir9fcOEWBeNbTY15n9Fw?e=xa2XKK).
- run pre_process_jsons to produce new json files which have a new entry with word-level alignments for each utterance.
- adjust the paths in configs/local_chime.yaml and use for the json paths the pre-processed jsons.
- run trainer.py

 