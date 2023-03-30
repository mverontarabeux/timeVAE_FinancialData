import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import  DataLoader

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math 

from utils import SequenceDataset, anderson_evaluation, kendall_evaluation
from utils import train_vae

class convVAE(nn.Module):
	
	def __init__(self, encod_hidden_layer_sizes, 
				z_dim, sequence_size, nb_features, n_channels=1, 
				use_GPU=False):
		super(convVAE, self).__init__()

		self.n_channels = n_channels
		self.sequence_size = sequence_size
		self.nb_features = nb_features
		self.filter = 3
		self.stride = 2
		self.padding = 2
		self.z_dim = z_dim
		self._gpu = use_GPU and torch.cuda.is_available()
		self.optimizer = None
		
		assert len(encod_hidden_layer_sizes) > 0
		self.encod_hidden_layer_sizes = encod_hidden_layer_sizes
		self.depth = len(self.encod_hidden_layer_sizes) - 1

		self.dim_flattened_height = math.ceil((self.sequence_size-self.filter+2*self.padding) / (self.stride*(len(self.encod_hidden_layer_sizes)))) + 1
		self.dim_flattened_width = math.ceil((self.nb_features-self.filter+2*self.padding) / (self.stride*(len(self.encod_hidden_layer_sizes)))) + 1

		self.max_dim_flattened = (self.encod_hidden_layer_sizes[-1] * 
			    					self.dim_flattened_height * 
									self.dim_flattened_width 
		)

		self.get_encoder_conv()
		self.get_decoder_conv()

		# move model to GPU if available
		if self._gpu:
			self.cuda()

	def get_encoder_conv(self):
		self.encode_1 = nn.Sequential()
		for i in range(len(self.encod_hidden_layer_sizes)-1):
			# Fully connected layer
			self.encode_1.append(nn.Conv2d(in_channels=self.encod_hidden_layer_sizes[i], 
				  							out_channels=self.encod_hidden_layer_sizes[i+1],
											  kernel_size=3,
											  stride=self.stride,
											  padding=self.padding))
			# Max pooling
			#self.encode_1.append(nn.MaxPool2d(2,2))
			# Batch normalization
			self.encode_1.append(nn.BatchNorm2d(self.encod_hidden_layer_sizes[i+1]))
			# Activation
			self.encode_1.append(nn.LeakyReLU())

		self.encode_2 = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.max_dim_flattened,100),
			nn.ReLU()
		)

		self.encode_mu = nn.Linear(100, self.z_dim)
		self.encode_log_var = nn.Linear(100, self.z_dim)

		# move encoder to GPU if available
		if self._gpu:
			self.encode_1.cuda()
			self.encode_2.cuda()
			self.encode_mu.cuda()
			self.encode_log_var.cuda()

	def encode(self, x):
		encode_1 = self.encode_1(x).cuda() if self._gpu else self.encode_1(x)
		encode_2 = self.encode_2(encode_1).cuda() if self._gpu else self.encode_2(encode_1)
		
		z_mu = self.encode_mu(encode_2).cuda() if self._gpu else self.encode_mu(encode_2)
		z_log_var = self.encode_log_var(encode_2).cuda() if self._gpu else self.encode_log_var(encode_2)
		# print(f"mean {torch.mean(z_mu)}")
		# print(f"std {torch.mean(z_log_var)}")
		return z_mu, z_log_var

	def get_decoder_conv(self):
		
		self.decode_1 = nn.Sequential(
			nn.Linear(self.z_dim,100),
			nn.ReLU(),
			nn.Linear(100,self.max_dim_flattened),
			nn.ReLU()
		)
	
		self.encod_hidden_layer_sizes.reverse()
		self.decode_2 = nn.Sequential()
		for i in range(len(self.encod_hidden_layer_sizes)-1):
			# Fully connected layer
			self.decode_2.append(nn.ConvTranspose2d(in_channels=self.encod_hidden_layer_sizes[i], 
					   						  out_channels=self.encod_hidden_layer_sizes[i+1],
											  kernel_size=self.filter,
											  stride=self.stride,
											  padding=self.padding,
											  output_padding=1))
			# Max pooling
			#self.decode_2.append(nn.MaxPool2d(2,2))
			# Batch normalization
			self.decode_2.append(nn.BatchNorm2d(self.encod_hidden_layer_sizes[i+1]))
			# Activation
			self.decode_2.append(nn.LeakyReLU())


		self.final_layer = nn.Conv2d(self.encod_hidden_layer_sizes[-1], 
		 							  out_channels=1, 
									  kernel_size=self.filter, 
									  padding=(0,1))
	
		# move layers to GPU if available
		if self._gpu:
			self.decode_1.cuda()
			self.decode_2.cuda()
			self.final_layer.cuda()


	def decode(self, z):
		z_decoded_1 = self.decode_1(z).cuda() if self._gpu else self.decode_1(z)
		z_decoded_1 = z_decoded_1.view(-1,
				 					   self.encod_hidden_layer_sizes[0],
									   self.dim_flattened_height,
									   self.dim_flattened_width)
		
		z_decoded_2 = self.decode_2(z_decoded_1).cuda() if self._gpu else self.decode_2(z_decoded_1)

		z_decoded_3 = self.final_layer(z_decoded_2).cuda() if self._gpu else self.final_layer(z_decoded_2)
		return z_decoded_3
	
		
	def reparameterize(self, mu, log_var):
		# this function samples a Gaussian distribution, with average (mu) and standard deviation specified (using log_var)
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		reparam =  (mu + eps*std).cuda()  if self._gpu else mu + eps*std
		return reparam # return z sample

	def forward(self, x):
		x = x.cuda() if self._gpu else x
		########################################
		# Pass forward in encoder
		########################################
		z_mu, z_log_var = self.encode(x)

		########################################
		# Sample
		########################################
		z_encoder_output = self.reparameterize(z_mu, z_log_var)
		
		########################################
		# Decode
		########################################
		x_decoded = self.decode(z_encoder_output)

		return z_mu, z_log_var, x_decoded

		
	def loss_function(self,x, x_decoded, mu, log_var):
		x = x.cuda() if self._gpu else x
		x_decoded = x_decoded.cuda() if self._gpu else x_decoded

		###########################
		# Reconstruction loss 
		###########################
		# mean reduce and sum
		weight_reconstruction = 1
		#reconstruction_error = (x_decoded-x).pow(2).sum(axis=2).sum(axis=1).mean()
		xmean = torch.mean(x, axis=2)
		xdecmean = torch.mean(x_decoded, axis=2)
		reconstruction_error = (xmean-xdecmean).pow(2).sum()
		 # add sum term 
		reconstruction_error += (x-x_decoded).pow(2).sum()

		###########################
		# Kulback Leiber loss 
		###########################

		KLD = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1).mean()
		#print(f"reconstruction = {reconstruction_error}")
		#print(f"KLD = {KLD}")
		ELBO = weight_reconstruction*reconstruction_error + KLD

		return ELBO, reconstruction_error, KLD


	def prior_sample(self, num_samples:int):
		z = torch.randn(num_samples,
						self.z_dim)
		z = z.cuda() if self._gpu else z
		samples = self.decode(z)
		return samples


	def reconstructe(self, x):
		if x.ndim==3:
			to_rec = x[None, :, :, :]
		else:
			to_rec = x.copy()	
		to_rec = to_rec.detach().cuda() if self._gpu else to_rec.detach()
		return self.forward(to_rec)[-1]
	



if __name__ == '__main__':
	from getdata import get_tickers_data, get_CAC40_tickers
	from datetime import datetime

	
	nb_stocks=2
	sequence_size=2**9
	
	internet = False
	if internet:
		# Set the window for returns extraction
		start_date = '2017-01-02'
		end_date = '2023-02-08'
		# load the returns for a certain number of stocks
		all_data = get_tickers_data(get_CAC40_tickers(nb_stocks), start_date, end_date)
		
	else:
		all_data = pd.read_csv("temp_stocks.csv")
		all_data = all_data.set_index(all_data.columns[0], drop=True).iloc[:,:nb_stocks]
		#all_data = scaler.fit_transform(all_data)

	# parametrize the VAE structure and training 
	z_dim = 40
	n_epochs = 100
	encode_dim = [1,5,10,20]

	vae_model = convVAE(encod_hidden_layer_sizes=encode_dim,
					sequence_size=sequence_size, 
					nb_features=all_data.shape[1], 
					z_dim=z_dim)

	vae_model.optimizer = optim.Adam(vae_model.parameters(), lr=1e-5)
								
	# Set the dataloader
	train = SequenceDataset(all_data, sequence_length=sequence_size)
	sequences_dataloader = DataLoader(train, batch_size=50, shuffle=True)

	retrain = True
	if retrain:
		# Launch the training
		train_vae(vae_model, 
				  sequences_dataloader,
				  n_epochs, 
				  dim4=True,
				  eval_data=all_data[:sequence_size])
		
		# Save the parameters
		date = datetime.now().strftime("%Y%m%d")
		torch.save(vae_model.state_dict(), f"trained_model\\vae_{date}")
	else:
		# date = "20230223"
		date = datetime.now().strftime("%Y%m%d")
		vae_model.load_state_dict(torch.load(f"trained_model\\vae_{date}"))
