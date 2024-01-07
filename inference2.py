import os
ROOT_PATH = os.getcwd()
import sys
WORK_PATH = f'{ROOT_PATH}/core/Fast_Wav2Lip/'
sys.path.append(WORK_PATH)

from os import listdir, path

import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time
import threading
import copy
import queue

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='./temp/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=8)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=16)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results

def load_cache(args, max_frames):
	face_pose = args.face.replace("video.mp4", "face_pose.npy")
	print(f"face_pose {face_pose}")
	face_det_results = np.load(face_pose, allow_pickle=True) 
	face_det_results = face_det_results[:max_frames]
	faces_origin = [face_det_result[0] for face_det_result in face_det_results]
	faces = [cv2.resize(face, (args.img_size, args.img_size)) for face in faces_origin]
	coors = [face_det_result[1] for face_det_result in face_det_results]
	return faces, coors

def datagen(faces, mels):
	idx_batch, img_batch, mel_batch = [], [], []
	for idx, m in enumerate(mels):
		idx_batch.append(idx)
		img_batch.append(faces[idx])
		mel_batch.append(m)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0
			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield idx_batch, img_batch, mel_batch
			idx_batch, img_batch, mel_batch = [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield idx_batch, img_batch, mel_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	model = model.to(device)
	return model.eval()

def load_melspetrogram(audio_path, fps = 30):
	wav = audio.load_wav(audio_path, 16000)
	mel = audio.melspectrogram(wav)
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
	full_mels = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while True:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			full_mels.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		else:
			full_mels.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1
	return full_mels

# load full frame 
frames_path ="core/Fast_Wav2Lip/data/lib/01/frames"
list_frames_path = glob(f"{frames_path}/*")
full_frames = []
for idx in range(1, len(list_frames_path)+1):
	frame_path = f"{frames_path}/{idx:04d}.png"
	frame = cv2.imread(frame_path)
	if args.resize_factor > 1:
		frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
	if args.rotate:
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
	
	# crop vung mat
	y1, y2, x1, x2 = args.crop
	if x2 == -1: x2 = frame.shape[1]
	if y2 == -1: y2 = frame.shape[0]
	frame = frame[y1:y2, x1:x2]
	full_frames.append(frame)
args.face = f"core/Fast_Wav2Lip/data/lib/01/video.mp4"
args.checkpoint_path = "core/Fast_Wav2Lip/checkpoints/wav2lip.pth" 
full_faces, full_coors = load_cache(args, len(full_frames))
frame_h, frame_w = full_frames[0].shape[:-1]


model = load_model(args.checkpoint_path)
def lipsync(face,coor, mel, img):
	img_batch, mel_batch = [face], [mel]
	img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
	img_masked = img_batch.copy()
	img_masked[:, args.img_size//2:] = 0
	img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
	mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

	img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
	mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
	
	with torch.no_grad():
		pred_batch = model(mel_batch, img_batch)

	pred_batch = pred_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
	pred = pred_batch[0]

	y1, y2, x1, x2 = coor
	pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
	img[y1:y2, x1:x2] = pred
	
	return img

def get_frame():
	i = -1
	step = 1
	max_frames = len(full_coors)
	while True:
		if i == max_frames - 1:
			step = -1
		elif i == 1:
			step = 1
		i = i + step
		print(f"i {i}")
		yield full_faces[i],full_coors[i], full_frames[i]
if __name__ == '__main__':
	lipsync(f"{WORK_PATH}/data/lib/01/video.mp4", f"{WORK_PATH}/data/test/10s/audio.wav")