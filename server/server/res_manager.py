from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt

from django.apps import AppConfig

import os
import numpy as np
import json
import re
from enum import Enum
import tensorflow as tf
from tensorflow.contrib import learn

from .forms import UploadFileForm

from scipy import misc
import cv2

import random

import time

def handle_uploaded_file(f, destinationPath):
	with open(destinationPath, 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

	return destinationPath 

def binarize(img):
	img[img<200] = 0
	img[img>200] = 1
	return img

def weightedAverage(pixel):
	return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def rgb2grey(image):
	if len(image.shape)==2:
		return image
	return cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

def threshold(img):
	threshold = np.average(img)
	
	img[img<threshold] = 0
	img[img>threshold] = 255

	return img

def preprocess_image(img_path):
	img = misc.imread(img_path)

	img_resized = misc.imresize(img,(28,28))
	img_bw = rgb2grey(img_resized)

	img_bw = threshold(img_bw)
	misc.imsave('../eval/img_'+str(time.time())+'.jpg', img_bw)

	img_binary = binarize(img_bw)

	return img_binary

def preprocess_maintain_aspect_ratio(img_path):
	img = misc.imread(img_path)
	imgray = rgb2grey(img)
	imgray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	
	#crop the main content
	ret,thresh = cv2.threshold(imgray,127,255,0)
	im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

	img_height, img_width = imgray.shape
	x_0_min, y_0_min = img_width, img_height
	x_1_max, y_1_max = 0, 0

	#get the best bounding box
	print(img_height, img_width)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		print(x,y,w,h)
		if x==0 and y==0 and w>0.9*img_width and h>0.9*img_height:
			continue
		if x < x_0_min:
			x_0_min = x
		if y < y_0_min:
			y_0_min = y
		if x+w > x_1_max:
			x_1_max = x+w
		if y+h > y_1_max:
			y_1_max = y+h

	if x_0_min>x_1_max or y_0_min>y_1_max:
		x_0_min = 0
		y_0_min = 0
		x_1_max = img_width
		y_1_max = img_height


	imgray = imgray[y_0_min:y_1_max,x_0_min:x_1_max]
	print(imgray.shape)

	#maintain the aspect ratio while resizing
	if w>h:
		w_resize = 28
		h_resize = int(h*28/w)
		padding_h = 28 - h_resize
		padding_w = 0
	else:
		h_resize = 28
		w_resize = int(w*28/h)
		padding_w = 28 - w_resize
		padding_h = 0

	img_resized = misc.imresize(imgray,(h_resize,w_resize))
	rows_in_img_resized, cols_in_img_resized = img_resized.shape
	img_final = np.ones((28,28))
	img_final *= 255

	y_start = int(random.random()*(28-rows_in_img_resized))
	y_end = y_start + rows_in_img_resized
	x_start = int(random.random()*(28-cols_in_img_resized))
	x_end = x_start + cols_in_img_resized
	img_final[y_start:y_end,x_start:x_end] = img_resized[:,:]

	# img_bw = threshold(img_final)
	misc.imsave('../evel/test.jpg', img_final)

	img_binary = binarize(img_final)

	return img_binary

class Ocr(AppConfig):


	def __init__(self):
		print("initialising module")

		self.checkpoint_dir = "/home/anson/Desktop/hackathons/sbi hackathon/ocr/ocr_mnist/models_large"
		self.allow_soft_placement = True
		self.log_device_placement = False

		self.saveDir = 'uploads/'

		self.labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','M','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		self.labels_order = json.load(open('./../labels_order.json', 'r'))

		checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
		graph = tf.Graph()
		with graph.as_default():
			session_conf = tf.ConfigProto(
			allow_soft_placement=self.allow_soft_placement,
			log_device_placement=self.log_device_placement)
			self.sess = tf.Session(config=session_conf)
			with self.sess.as_default():
				# Load the saved meta graph and restore variables
				saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
				saver.restore(self.sess, checkpoint_file)

				# Get the placeholders from the graph by name
				self.input_x = graph.get_operation_by_name("input_x").outputs[0]
				self.dropout_keep_prob = graph.get_operation_by_name("dropout/keep_prob").outputs[0]

				# Tensors we want to evaluate
				self.scores = graph.get_operation_by_name("fc2/yconv").outputs[0]


	@csrf_exempt
	def getCharacter(self, request):

		if request.method == "POST":

			form = UploadFileForm(request.POST, request.FILES)
			fileName = self.saveDir + request.FILES['file'].name
			handle_uploaded_file(request.FILES['file'], fileName)

			x_img = [preprocess_image(fileName)]
			x_test = np.array(x_img, dtype=np.float32)

			batch_scores = self.sess.run(self.scores, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
			batch_scores_digits = batch_scores[:,0:10]
			batch_scores_alphabets = batch_scores[:,10:]

			std_dev = np.std(batch_scores[0])
			print("std_dev")
			print(std_dev)
			
			char_scores = {}
			for i, label in enumerate(self.labels):
				char_scores[label] = str(batch_scores[0][i])

			detected_char =  self.labels[int(self.labels_order[np.argmax(batch_scores[0])][-3:])] if batch_scores[0,np.argmax(batch_scores[0])]>9 else " "
			detected_digit = self.labels[int(self.labels_order[np.argmax(batch_scores_digits[0])][-3:])] if batch_scores_digits[0,np.argmax(batch_scores_digits[0])>9] else " "
			detected_alphabet = self.labels[int(self.labels_order[10+np.argmax(batch_scores_alphabets[0])][-3:])] if batch_scores_alphabets[0,np.argmax(batch_scores_alphabets[0])]>9 else " "

			res = {
				'scores': char_scores,
				'detected_char': detected_char,
				'detected_digit': detected_digit,
				'detected_alphabet': detected_alphabet,
			}

			print("res")
			print(res)
		

		return JsonResponse(res,safe=False)

