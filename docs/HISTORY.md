# History

How the voice-matching tool came about, in order. Newest entries go at the bottom.

## Background: the ORYZN brand film (8–9 October 2026)

- **8 Oct.** A 30-second motion-graphics brand film for ORYZN was attempted on HeyGen HyperFrames. The renders had invisible text, the wrong fonts and no audio, and the account's three free renders ran out. The film was rebuilt as an HTML composition rendered locally with headless Chromium and FFmpeg.
- **9 Oct.** Following a reference video (a narrated Bangla brand story), an 82-second illustrated story film was made: the "Made in Bangladesh" label, Mughal Dhaka, muslin, jamdani, the first garment export in 1978, then ORYZN with real product photos and the ORYZN logo. Bangla word-by-word captions and an original score with sound design. The historical facts were checked against sources (Banglapedia, The Daily Star, TBS News, Jago News 24), and the export claim was corrected because Vietnam passed Bangladesh in 2025.

## The voiceover (9 October 2026)

1. Asked whether a voice could be generated. Answer: free Bangla text-to-speech would sound robotic; better to record a real read, or use ElevenLabs.
2. Asked whether a recording could be matched to a reference voice instead. Answer: yes, with speech-to-speech voice conversion, on condition that the reference voice's owner agrees.
3. Received the team member's read of the script (75.8 s). While waiting for the reference, the read was aligned to the 23 caption lines, and the film was re-timed to follow the voice.
4. Received the reference voice (37.6 s). A speaker check showed it isn't the narrator from the reference ad; asked whose voice it is, and the answer was "mine or a team member's".
5. Experiments (see [RESEARCH.md](RESEARCH.md)): three converters with and without a Praat pre-shift, then a pre-shift sweep. OpenVoice v2 after the pre-shift won.
6. All 23 lines were converted with six settings each, and the best take per line was kept (similarity 0.90, char errors 16.7%). The voice was mixed over the ducked score at −14 LUFS and delivered as `ORYZN_Story_VO_9x16.mp4` (1:24).

## Packaging the tool (10 October 2026)

1. Asked for the technique, so voices could be matched without coming back each time, and whether to build a tool for it. Answer: yes, starting from the exact pipeline, packaged.
2. **The cloud machine was reset before the package was finished.** Lost: the working folders, the Python environment, the ORYZN story film's source (scenes, animation code, score generator) and the voiceover scripts. What survived was everything that had been sent in the conversation: the delivered videos and caption file on the user's side, plus the recordings, logo, photos and the early 30-second film's code, which came back as re-attached files.
3. The tool was rewritten as `voice_match.py` and tested from a clean install on the same two recordings.
4. Fixes found by testing: a peak limiter (the output was landing at −19 LUFS instead of −16), M4A input via FFmpeg, and loading the Bangla recogniser in one format only (1.2 GB instead of 2.5 GB).
5. Final check: the README's commands, run word for word in a fresh folder with empty model caches. Install took 60 s. Fast mode took 56 s (similarity 0.90, char errors 18.8%); `--best-of` took 6.5 min including downloads (0.91, 15.3%). The output was −16.0 LUFS with −0.9 dBFS peaks. That run caught the recogniser downloading twice; it was fixed and re-tested.
6. This repository was created so that a machine reset can't lose the work again.

## How to keep this history useful

- Every change to the tool goes in [CHANGELOG.md](../CHANGELOG.md).
- New experiments and their numbers go in [RESEARCH.md](RESEARCH.md), measured with `tools/evaluate.py` so they compare with the old ones.
- Recordings of real voices stay out of this public repository.
