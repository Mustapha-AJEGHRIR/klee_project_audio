# LibriCount10 0dB Dataset

This is the description to the LibriCount10 synthetic dataset for speaker count estimation. 

Therefore for each recording we provide the ground truth number of speakers within the file name, where `k` in, `k_uniquefile.wav` is the maximum number of concurrent speakers with the 5 seconds of recording.

The dataset contains a simulated cocktail party environment of [0..10] speakers, mixed with 0dB SNR from random utterances of different speakers from the [LibriSpeech](http://www.openslr.org/12/) `CleanTest` dataset. 

All recordings are of 5s durations, and all speakers are active for the most part of the recording.

For each unique recording, we provide the audio wave file (16bits, 16kHz, mono) and an annotation `json` file with the same name as the recording.

## Metadata

In the annotation file we provide information about the speakers sex, their unique speaker_id, and vocal activity within the mixture recording in samples. Note that these were automatically generated using [a voice activity detection method](https://github.com/wiseman/py-webrtcvad).

In the following example a speaker count of 3 speakers is the ground truth.

```json
[
	{
		"sex": "F", 
		"activity": [[0, 51076], [51396, 55400], [56681, 80000]], "speaker_id": 1221
	}, 
	{
		"sex": "F", 
		"activity": [[0, 51877], [56201, 80000]], 
		"speaker_id": 3570
	}, 
	{
		"sex": "M", 
		"activity": [[0, 15681], [16161, 68213], [73498, 80000]], "speaker_id": 5105
	}
]
```