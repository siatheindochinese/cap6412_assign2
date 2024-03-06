import torch
from PIL import Image
import numpy as np
import os
import json
import csv
import random

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

device='cuda'

################
# Load Dataset #
################
YOUR_LOCAL_PATH = 'datasets/WGV'

if 'opt_closedvqa_result.json' not in os.listdir('precomputed_wgv'):
	model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
###########################
# Preprocess data for VQA #
###########################
#preprocess annotations
if 'cityToCountry.json' not in os.listdir(YOUR_LOCAL_PATH):
	geoannot = []
	cityToCountry = {}
	line_count = 0
	countries = []
	with open(YOUR_LOCAL_PATH +'/labels_list.csv') as f:
		csv_reader = csv.reader(f, delimiter=',')
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				city = row[0]
				country = row[2]
				if city not in cityToCountry:
					cityToCountry[city] = country
				if country not in countries:
					countries.append(country)
				line_count += 1
	print(f'Processed {line_count} lines.')
	with open(YOUR_LOCAL_PATH + '/cityToCountry.json','w') as f:
		json.dump(cityToCountry, f)
	with open(YOUR_LOCAL_PATH + '/countries.json','w') as f:
		json.dump(countries, f)

cityToCountry = json.load(open(YOUR_LOCAL_PATH + '/cityToCountry.json'))
countries = json.load(open(YOUR_LOCAL_PATH + '/countries.json'))
	
#preprocess image gt
imgpth = os.listdir(YOUR_LOCAL_PATH+'/val')

citygt = ['_'.join(ele.split('_')[:-2]) for ele in imgpth]
countrygt = [cityToCountry[city] for city in citygt]

imgpth = [YOUR_LOCAL_PATH+'/val/'+ele for ele in imgpth]

#preprocess text prompts
template = '({}) Question: Which of these countries is this image from? Answer:'

#############################
# Closed-ended VQA Pipeline #
#############################
# 4 choice multiple-choice question-answering
result = []
if 'opt_closedvqa_result.json' not in os.listdir('precomputed_wgv'):
	for i in range(len(imgpth)):
		raw_image = Image.open(imgpth[i]).convert("RGB")
		image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
		
		context = random.sample(countries,3)
		context.append(countrygt[i])
		random.shuffle(context)
		
		ques = template.format('/'.join(context))
		text_processed = txt_processors["eval"](ques)
		print(text_processed)
		ans = model.predict_answers(samples={"image": image_processed, "text_input": text_processed}, inference_method="generate", min_len=2)
		print(ans)
		result.append(ans[0])
	with open('precomputed_wgv/opt_closedvqa_result.json','w') as f:
		json.dump(result, f)
		
result = json.load(open('precomputed_wgv/opt_closedvqa_result.json'))

cities = list(cityToCountry.keys())

for index, res in enumerate(result):
	resori = res
	if len(res.split(' ')) > 1:
		res = res.split(' ')
		res = '_'.join(res)
		
	for country in countries:
		if country.lower() in res:
			result[index] = country.lower()
			
	for city in cities:
		if city.lower() in res:
			result[index] = cityToCountry[city].lower()
			
tally = 0
for index, gt in enumerate(countrygt):
	print(index)
	print('pred:', result[index])
	print('gt:', gt.lower())
	print('')
	if result[index] == gt.lower():
		tally +=1

accuracy = tally/len(result)
print('accuracy:', round(accuracy * 100,1), '%')
