# AMI starter subset attribution

Original dataset: AMI Meeting Corpus, AMI Consortium.

Official corpus: https://groups.inf.ed.ac.uk/ami/corpus/
Official downloads: https://groups.inf.ed.ac.uk/ami/download/
Manual annotation archive: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
Source WAVs: official AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Mix-Headset.wav.

License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated in the original archive's LICENCE.txt, copied alongside this file as LICENCE_AMI.txt. https://creativecommons.org/licenses/by/4.0/

Changes: contiguous audio ranges were selected from the original PCM WAVs; short registration references were cropped outside the evaluation intervals. Original conversation order, voices and channel layout were preserved. No voice was removed or synthesized. Annotation subsets were converted from NXT XML into JSON/JSONL and an automatic candidate review table.

Per-file source URLs, frame ranges, hashes and annotation IDs are recorded in selected_candidates.json, enrollment_manifest.json, source_audio_headers.json, and the parent download_manifest.json. Automatic question/response event selection is not yet verified by human listening. This pilot subset is not a final benchmark and contains no model performance results.
