# Voice Companion

A local, real-time **voice-to-voice AI companion** built on [Sesame CSM-1B](https://github.com/SesameAILabs/csm). Talk to it in your browser and it answers out loud, in a voice cloned from a short sample, with about 0.1 s to first audio. You can interrupt it mid-sentence.

- **Hands-free conversation:** voice activity detection works out when you've finished talking (Silero VAD + faster-whisper).
- **Natural interruptions:** say "stop" or "wait" to cut it off. Acknowledgements like "yeah" or "mhm" don't interrupt.
- **Streaming speech:** CSM audio is generated frame by frame and streamed to the browser. A `torch.compile`d decoder tunes itself to your GPU.
- **Steadier cloned voice:** short sentences are merged, each sentence is conditioned on the previous one, and rushed sentences are slowed back to the reference voice's natural pace (pitch-preserving).
- **Memory:** it remembers facts about you between conversations, stored locally in SQLite.
- **Debug panel:** for every sentence, compare the text CSM was asked to say with what was actually heard, with the pitch curve against the reference voice.
- **Any LLM:** works with llama.cpp, Ollama, LM Studio, or any OpenAI-compatible server.

Everything runs locally. **See [INSTALL.md](INSTALL.md) to get started.**

## Responsible use

Voice cloning can be misused. Only clone voices you have the right to use, with the speaker's explicit consent. Sesame's CSM terms prohibit mimicking real individuals without their consent, impersonation or fraud, misinformation or deception, and illegal or harmful activities.

## Licence

Apache License 2.0. See [LICENSE](LICENSE). Includes Sesame's CSM reference code (Apache 2.0) in `csm/`; see [NOTICE](NOTICE) for third-party credits.
