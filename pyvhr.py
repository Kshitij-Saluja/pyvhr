from pyVHR.signals.video import Video

# -- Video object
videoFilename = 'VideoStreamOutput.avi'

video = Video(videoFilename)

# -- extract faces
video.getCroppedFaces(detector='mtcnn', extractor='opencv')
video.printVideoInfo()

# -- apply remote PPG method
from pyVHR.methods.ssr import SSR

params = {"video": video, "verb":0, "ROImask":"skin_adapt", "skinAdapt":0.2}

ssr = SSR(**params)

# -- get BPM values
bpmES_ssr, timesES_ssr = ssr.runOffline(**params)
print(bpmES_ssr)