import torch
from PIL import Image
import numpy as np
import os
import json
import csv

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

device='cuda'

################
# Load Dataset #
################
YOUR_LOCAL_PATH = 'datasets/WGV'

#############################
# Load Model and Processors #
#############################
'''
model, vis_processors, txt_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
'''

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

######################################
# Preprocess data for retrieval task #
######################################
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
templates = ['a photo i took in {}.',
			 'a photo i took while visiting {}.',
			 'a photo from my home country of {}.',
			 'a photo from my visit to {}.',
			 'a photo showing the country of {}.']

##############################################
# Generate ITC score matrix for all templates#
##############################################
'''
We follow Section 4.4 of the Blip2 paper which is inspired by ALBEF.
ITM is done on the top k=128 ITC matches for t2i and i2t.
According to the paper, this is done to save memory and computation.
We k=10 (as opposed to k= 91) due to memory and time constraints.
'''
images_embedding = None
#template 0
itc0 = np.zeros((len(imgpth),len(countries)))

if 'itc0.txt' not in os.listdir('precomputed_wgv'):
	if images_embedding is not None:
		images_embedding = images_embedding
	else:
		images_embedding = None
		# extract image embeddings
		for i in range(len(imgpth)):
			print('i:',i)
			raw_image = Image.open(imgpth[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			sample = {"image": image_processed, "text_input": None}
			image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
			if images_embedding is None:
				images_embedding = image_emb
			else:
				images_embedding = torch.cat((images_embedding, image_emb),0)
	
	texts_embedding = None

	# extract text embeddings
	for i in range(len(countries)):
		print('t:',i)
		caption = templates[0].format(countries[i])
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc0 = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_wgv/itc0.txt', itc0)
else:
	itc0 = np.loadtxt('precomputed_wgv/itc0.txt')
	
#template1
if 'itc1.txt' not in os.listdir('precomputed_wgv'):
	if images_embedding is not None:
		images_embedding = images_embedding
	else:
		images_embedding = None
		# extract image embeddings
		for i in range(len(imgpth)):
			print('i:',i)
			raw_image = Image.open(imgpth[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			sample = {"image": image_processed, "text_input": None}
			image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
			if images_embedding is None:
				images_embedding = image_emb
			else:
				images_embedding = torch.cat((images_embedding, image_emb),0)
	
	texts_embedding = None

	# extract text embeddings
	for i in range(len(countries)):
		print('t:',i)
		caption = templates[1].format(countries[i])
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc1 = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_wgv/itc1.txt', itc1)
else:
	itc1 = np.loadtxt('precomputed_wgv/itc1.txt')
	
#template2
if 'itc2.txt' not in os.listdir('precomputed_wgv'):
	if images_embedding is not None:
		images_embedding = images_embedding
	else:
		images_embedding = None
		# extract image embeddings
		for i in range(len(imgpth)):
			print('i:',i)
			raw_image = Image.open(imgpth[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			sample = {"image": image_processed, "text_input": None}
			image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
			if images_embedding is None:
				images_embedding = image_emb
			else:
				images_embedding = torch.cat((images_embedding, image_emb),0)
	
	texts_embedding = None
	# extract text embeddings
	for i in range(len(countries)):
		print('t:',i)
		caption = templates[2].format(countries[i])
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc2 = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_wgv/itc2.txt', itc2)
else:
	itc2 = np.loadtxt('precomputed_wgv/itc2.txt')
	
#template3
if 'itc3.txt' not in os.listdir('precomputed_wgv'):
	if images_embedding is not None:
		images_embedding = images_embedding
	else:
		images_embedding = None
		# extract image embeddings
		for i in range(len(imgpth)):
			print('i:',i)
			raw_image = Image.open(imgpth[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			sample = {"image": image_processed, "text_input": None}
			image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
			if images_embedding is None:
				images_embedding = image_emb
			else:
				images_embedding = torch.cat((images_embedding, image_emb),0)
	
	texts_embedding = None

	# extract text embeddings
	for i in range(len(countries)):
		print('t:',i)
		caption = templates[3].format(countries[i])
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc3 = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_wgv/itc3.txt', itc3)
else:
	itc3 = np.loadtxt('precomputed_wgv/itc3.txt')
	
#template4
if 'itc4.txt' not in os.listdir('precomputed_wgv'):
	if images_embedding is not None:
		images_embedding = images_embedding
	else:
		images_embedding = None
		# extract image embeddings
		for i in range(len(imgpth)):
			print('i:',i)
			raw_image = Image.open(imgpth[i]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			sample = {"image": image_processed, "text_input": None}
			image_emb = model.extract_features(sample, mode="image").image_embeds[:,0,:]
			if images_embedding is None:
				images_embedding = image_emb
			else:
				images_embedding = torch.cat((images_embedding, image_emb),0)
	
	texts_embedding = None

	# extract text embeddings
	for i in range(len(countries)):
		print('t:',i)
		caption = templates[4].format(countries[i])
		text_processed = txt_processors["eval"](caption)
		sample = {"image": None, "text_input": text_processed}
		text_emb = model.extract_features(sample, mode="text").text_embeds[:,0,:]
		if texts_embedding is None:
			texts_embedding = text_emb
		else:
			texts_embedding = torch.cat((texts_embedding, text_emb),0)

	# normalize each tensor
	images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
	texts_embedding /= texts_embedding.norm(dim=-1, keepdim=True)

	itc4 = images_embedding.cpu().detach().numpy() @ texts_embedding.cpu().detach().numpy().T
	np.savetxt('precomputed_wgv/itc4.txt', itc4)
else:
	itc4 = np.loadtxt('precomputed_wgv/itc4.txt')

#############################
# Generate ITM score matrix #
#############################
#stack all itc scores
itc = np.concatenate((itc0,itc1,itc2,itc3,itc4),axis=0)

k = 10
# filter top k=10 for i2t
# i2t, matrix of (n(i)x5) x 10
# elements are the top k=10 text indices
if 'i2t_idx.txt' not in os.listdir('precomputed_wgv'):
	i2t_idx = np.argpartition(itc, -k)[:,-k:]
	np.savetxt('precomputed_wgv/i2t_idx.txt', i2t_idx)
else:
	i2t_idx = np.loadtxt('precomputed_wgv/i2t_idx.txt').astype(int)

# Load ITM model
model, vis_processors, txt_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

itm_i2t = np.zeros((len(imgpth)*5, k))
if 'itm_i2t.txt' not in os.listdir('precomputed_wgv'):
	for i in range(len(imgpth)*5):
		for j in range(k):
			print(i,j)
			t  = i2t_idx[i][j]
			
			i_ori = i % len(imgpth)
			raw_image = Image.open(imgpth[i_ori]).convert("RGB")
			image_processed = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
			
			template_idx = i // len(imgpth)
			caption = templates[template_idx].format(countries[t])
			text_processed = txt_processors["eval"](caption)
			
			itm_output = model({"image": image_processed, "text_input": text_processed}, match_head="itm")
			itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
			itm_i2t[i][j] = itm_scores[:, 1].item()
	np.savetxt('precomputed_wgv/itm_i2t.txt', itm_i2t)
else:
	itm_i2t = np.loadtxt('precomputed_wgv/itm_i2t.txt')
	
# get zero-shot accuracy from top 1 result
tally = 0
for index, score in enumerate(itm_i2t):
	print('i:', index)
	top1 = np.argmax(score)
	top1 = i2t_idx[index][top1]
	pred = countries[top1]
	
	index = index % len(imgpth)
	gt = countrygt[index]
	print('pred:', pred)
	print('gt:', gt)
	print('')
	if pred == gt:
		tally +=1
accuracy = tally / (len(imgpth) * 5)
print('zero-shot country classification accuracy: ', round(accuracy * 100,1), '%')
