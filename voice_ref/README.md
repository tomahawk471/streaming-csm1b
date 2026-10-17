# voice_ref/

The voice the companion speaks with is cloned from two files here:

- `sample.wav`: 15-45 seconds of one person speaking: clean, calm and conversational, with no music or background noise.
- `sample.txt`: the **exact** words spoken in `sample.wav`.

Get a public-domain default with `python scripts/download_default_voice.py`, or see INSTALL.md, "Using your own voice".

Only use a voice you have the rights to and the speaker's consent for. These files are git-ignored, so they never get committed.
