import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import  DataLoader

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from utils import SequenceDataset, anderson_evaluation, kendall_evaluation
from utils import train_vae, plot_eval, eval_VAE

class denseVAE(nn.Module):
	def __init__(self, encod_hidden_layer_sizes, 
				decod_hidden_layer_sizes,
				z_dim, sequence_size, nb_features, n_channels=1, 
				conv=False,
				use_GPU=False):
		super(denseVAE, self).__init__()

		self.n_channels = n_channels
		self.sequence_size = sequence_size
		self.nb_features = nb_features
		self.input_dim = (self.sequence_size)*(self.nb_features)
		self.z_dim = z_dim
		self._gpu = use_GPU and torch.cuda.is_available()
		self.optimizer = None

		assert len(encod_hidden_layer_sizes) > 0 and len(decod_hidden_layer_sizes) > 0
		self.encod_hidden_layer_sizes = encod_hidden_layer_sizes
		self.decod_hidden_layer_sizes = decod_hidden_layer_sizes

		self.get_encoder()
		self.get_decoder()

		# move model to GPU if available
		if self._gpu:
			self.cuda()

	def get_encoder(self):
		########################################
		# Define the main network for mu and var
		########################################
		self.encode_main = nn.Sequential(nn.Flatten(start_dim=1))
		previous_dim = self.input_dim          # !! input dim 
		for i, nbunit in enumerate(self.encod_hidden_layer_sizes, start=1):
			# Fully connected layer
			self.encode_main.append(nn.Linear(previous_dim, nbunit))
			# Regularisation
			if nbunit>=100:
				self.encode_main.append(nn.Dropout(0.2))
			# Activation
			#if i!=len(self.encod_hidden_layer_sizes):
			self.encode_main.append(nn.ReLU())
			previous_dim = nbunit

		########################################
		# Put that in the sequential api with dense output for each
		########################################
		self.encode_mu = nn.Linear(previous_dim, self.z_dim)
		self.encode_log_var = nn.Linear(previous_dim, self.z_dim)

		# move encoder to GPU if available
		if self._gpu:
			self.encode_main.cuda()
			self.encode_mu.cuda()
			self.encode_log_var.cuda()


	def encode(self, x):
		result = self.encode_main(x).cuda() if self._gpu else self.encode_main(x)
		z_mu = self.encode_mu(result).cuda() if self._gpu else self.encode_mu(result)
		z_log_var = self.encode_log_var(result).cuda() if self._gpu else self.encode_log_var(result)
		# print(f"mean {torch.mean(z_mu)}")
		# print(f"std {torch.mean(z_log_var)}")
		return z_mu, z_log_var

	def get_decoder(self):
		########################################
		# Define the main network
		########################################
		self.decode_main = nn.Sequential()
		previous_dim = self.z_dim              # !!z_dim 
		for i, nbunit in enumerate(self.decod_hidden_layer_sizes, start=1):
			# Fully connected layer
			self.decode_main.append(nn.Linear(previous_dim, nbunit))
			# Regularisation
			if nbunit>=100:
				self.decode_main.append(nn.Dropout(0.2))
			# Activation
			self.decode_main.append(nn.ReLU())
			previous_dim = nbunit
		
		self.decode_main.append(nn.Linear(previous_dim, self.input_dim)) # !!output_dim 

		# move encoder to GPU if available
		if self._gpu:
			self.decode_main.cuda()


	def decode(self, z):
		z_decoded = self.decode_main(z).cuda() if self._gpu else self.decode_main(z)
		z_decoded = z_decoded.view(-1,
									self.sequence_size, 
									self.nb_features)
		return z_decoded

		
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
		to_rec = x.cuda() if self._gpu else x
		return self.forward(to_rec)[-1]
	

if __name__ == '__main__':
	from getdata import get_tickers_data, get_CAC40_tickers
	from datetime import datetime

	
	nb_stocks=2
	sequence_size=300
	
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
	sequence_size = 300 
	z_dim = 40
	n_epochs = 100
	encode_dim = [200,50]
	decode_dim = [50,200,sequence_size*nb_stocks] # avant seulement sequence_size ???

	vae_model = denseVAE(encod_hidden_layer_sizes=encode_dim,
					decod_hidden_layer_sizes=decode_dim,
					sequence_size=sequence_size, 
					nb_features=all_data.shape[1], 
					z_dim=z_dim,
					conv=True)

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
				  all_data[:sequence_size])
			
		# Save the parameters
		date = datetime.now().strftime("%Y%m%d")
		torch.save(vae_model.state_dict(), f"trained_model\\vae_{date}")
	else:
		# date = "20230223"
		date = datetime.now().strftime("%Y%m%d")
		vae_model.load_state_dict(torch.load(f"trained_model\\vae_{date}"))
