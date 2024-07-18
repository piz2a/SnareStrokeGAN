# SnareStrokeGAN
GAN model which synthesizes snare drum stroke waves from MIDI by sorting and filtering single stroke samples.

### TODO:
- [X] Slicing / Annotating sound samples
- [X] Creating Embedding layer
- [X] Data Augmentation / Custom Dataset
- [X] Generator
- [ ] Discriminator
- [ ] Training (with CUDA) (Fix memory allocation exceed)
- [ ] Integrating with VST3 plugin

### How to use:
```sh
pip install -r requirements.txt
pip install ffmpeg-downloader
ffdl install --add-path
```
