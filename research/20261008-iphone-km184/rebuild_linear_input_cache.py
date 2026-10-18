"""Restore mono48k caches and decoded iPhone stereo for direct FIR experiment.
Requires imageio-ffmpeg; optional temporary package path follows prior environment.
Preserves original files. Decoder reports a trailing ALAC packet warning; valid
frames checked against source metadata (92909568) below.
"""
from pathlib import Path
import sys;sys.path.insert(0,"/private/tmp/mic-analysis-deps")
import numpy as np,soundfile as sf,subprocess,imageio_ffmpeg
src=Path('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10');out=Path('/private/tmp/mic-audit-20261008');out.mkdir(exist_ok=True)
for d,n in [('km184','neumann km184 guitar.wav'),('s21','s21 guitar.wav'),('iphone','iphone guitar.m4a')]:
 target=out/(d+'-guitar.npy')
 if target.exists():continue
 path=src/n
 if d=='iphone':
  path=out/'iphone-decoded-stereo.wav';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(src/n),'-ar','48000','-c:a','pcm_f32le',str(path)],check=True)
 info=sf.info(path);assert info.samplerate==48000
 if d=='iphone':assert info.frames==92909568,'Decoded valid frame count differs from original metadata'
 a=np.lib.format.open_memmap(target,mode='w+',dtype='float32',shape=(info.frames,));pos=0
 with sf.SoundFile(path) as f:
  for x in f.blocks(blocksize=720000,dtype='float32',always_2d=True):a[pos:pos+len(x)]=x.mean(axis=1);pos+=len(x)
 a.flush();print(d,pos,flush=True)
