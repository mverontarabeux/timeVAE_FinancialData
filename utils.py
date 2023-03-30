import numpy as np 
import pandas as pd
import torch
import math
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp 

class SequenceDatasetDisjoint(Dataset):
    # Cut distinct sequences from the passed data X 
    # Meaning a data point will be present only once
    # It is not rolling window sampling
    def __init__(self, X, y=None,sequence_length=64, gpu=False):
        self.sequence_length = sequence_length
        if y is None:
          self.y = torch.ones((X.shape[0],1))
        else:
          self.y = torch.tensor(y)
        
        if isinstance(X, pd.DataFrame):
          X = X.values

        self.X = torch.tensor(X)
        self.gpu = gpu

    def __len__(self):
        return math.ceil(self.X.shape[0]/self.sequence_length)

    def __getitem__(self, i): 
        if (i+1) <= self.X.shape[0]//self.sequence_length:
            x = self.X[(i * self.sequence_length):((i + 1)*self.sequence_length),:]
            y = self.y[(i * self.sequence_length):((i + 1)*self.sequence_length),:]
        else:
            # The last sequence to handle : padd the last line
            x = self.X[(i * self.sequence_length):, :]
            padding_x = torch.tensor(([np.asarray(x[-1])] * (self.sequence_length-x.shape[0])))
            x = torch.cat((x,padding_x), 0)
            y = self.y[(i * self.sequence_length):].squeeze()
            if self.gpu:
                padding_y = torch.tensor([int(y[-1])]*(self.sequence_length-len(y))).cuda()
            else:
                padding_y = torch.tensor([int(y[-1])]*(self.sequence_length-len(y)))
            y = torch.cat((y,padding_y))
            y = y[:,None]
        return x, y

class SequenceDataset(Dataset):
    # Cut distinct sequences from the passed data X 
    # Meaning a data point will be present only once
    # It is not rolling window sampling
    def __init__(self, X, y=None,sequence_length=64, gpu=False):
        self.sequence_length = sequence_length
        if y is None:
          self.y = torch.ones((X.shape[0],1))
        else:
          self.y = torch.tensor(y)
        
        if isinstance(X, pd.DataFrame):
          X = X.values

        self.X = torch.tensor(X).to(torch.float32)
        self.gpu = gpu

    def __len__(self):
        return math.ceil(self.X.shape[0] - self.sequence_length + 1)

    def __getitem__(self, i): 
        x = self.X[i:i+self.sequence_length,:]
        y = self.y[i:i+self.sequence_length,:]
        return x, y
    

def kl_anneal_function(anneal_function, step, k, x0):
	if anneal_function == 'logistic':
		return float(1/(1+np.exp(-k*(step-x0))))
	elif anneal_function == 'linear':
		return min(1, step/x0)
                

def train_step_vae(vae_model,
                   data_train_loader,
                   epoch,
                   midpoint,
                   grad_norm=1.0,
                   k=0.025,
                   dim4=False, 
                   anneal_function="logistic"):
        
	batch_loss = []
	batch_KLD = []
	batch_rec = []
	batch_KL_weight = []
	step_weight = epoch * len(data_train_loader)
	for batch_idx, (data, _) in enumerate(data_train_loader, start=1):
		if dim4:
			data = data[:, None, :, :]
		
		vae_model.optimizer.zero_grad()
		z_mu, z_log_var, x_decoded = vae_model(data) 

		_, reconstruction_error, KLD = vae_model.loss_function(data, x_decoded, z_mu, z_log_var) 

		####################################
		# ANNEALING FUNCTION FOR THE KL TERM
		####################################
		step_weight += 1
		KL_weight = 1
		if anneal_function is not None:
			KL_weight = kl_anneal_function(anneal_function, step_weight, k, midpoint)
		loss_vae = (KL_weight * KLD + reconstruction_error)/len(data)

		##################
		# BACK PROPAGATION
		##################
		loss_vae.backward()

		###################
		# GRADIENT CLIPPING
		###################
		torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=grad_norm)

		################
		# PARAMS UPDATES
		################            
		vae_model.optimizer.step() 

		################
		# DEBUGGING
		################       
		batch_loss.append(loss_vae.item())
		batch_rec.append(reconstruction_error.item())
		batch_KLD.append(KLD.item())
		batch_KL_weight.append(KL_weight)
			
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(data_train_loader.dataset),
			100. * batch_idx / len(data_train_loader), loss_vae.item() / len(data)))
	
	if epoch % 20 == 0:
		print('====> Epoch: {} Av. loss: {:.4f} Av. KLD: {:.4f} Av. rec: {:.4f}'.format(epoch, 
                                                                                  np.mean(batch_loss),
                                                                                  np.mean(batch_KLD),
                                                                                  np.mean(batch_rec)))

	return z_mu, z_log_var, batch_KLD, batch_rec, batch_KL_weight


def train_vae(vae_model, sequences_dataloader, n_epochs, 
                   grad_norm=1.0,
                   k=0.025,dim4=False, eval_data=None, 
                   anneal_function="logistic"):
	# Launch the training
	list_z_mu = []
	list_z_logvar = []
	list_ad = []
	list_ke = []
	list_KLD = []
	list_rec = []
	list_weights = []
	midpoint = (n_epochs*len(sequences_dataloader))//3
	all_reconstructions = []
	for i, epoch in enumerate(range(n_epochs+2), start=0):
		z_mu, z_log_var, batch_KLD, batch_rec, batch_KL_weight = train_step_vae(vae_model=vae_model, 
                                                                          data_train_loader=sequences_dataloader, 
                                                                          epoch=i, 
                                                                          midpoint=midpoint,
                                                                            grad_norm=grad_norm,
                                                                          k=k, 
                                                                          dim4=dim4, 
                                                                          anneal_function=anneal_function)
                
		list_z_mu.append(np.mean(z_mu.cpu().detach().numpy()))
		list_z_logvar.append(np.mean(z_log_var.cpu().detach().numpy()))
	
		list_KLD.append(np.mean(batch_KLD))
		list_rec.append(np.mean(batch_rec))
		list_weights.append(np.mean(batch_KL_weight))
        
		if not eval_data is None:
			vae_model.eval()
			assert len(eval_data) == vae_model.sequence_size
                        
			ad, ke, reconstructed_data = eval_VAE(vae_model, eval_data)
			all_reconstructions.append(reconstructed_data)
			if epoch % 20 == 0:
				plot_eval(reconstructed_data, eval_data)
			list_ad.append(ad)
			list_ke.append(ke)
                        
			vae_model.train()
			
	return list_z_mu, list_z_logvar, list_ad, list_ke, list_KLD, list_rec, list_weights, all_reconstructions


def plot_eval(reconstructed_data, data):
    assert isinstance(reconstructed_data, np.ndarray)
    assert isinstance(data, pd.DataFrame)

    # Reconstructe the data
    if reconstructed_data.ndim == 1:
        reconstructed_data = reconstructed_data[:, None]

    nb_to_plot = data.shape[1]
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    colors = ["dodgerblue", "orange"]

    # Calculate the number of rows and columns for subplots
    ncols = 2
    nrows = int(np.ceil(nb_to_plot / ncols))

    # Create a figure and set its size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 2*nrows))

    for i in range(nb_to_plot):
        # Flatten the axes array and get the current axis
        ax = axes.flatten()[i]

        bins = np.linspace(min(data.iloc[:, i].values),
                            max(data.iloc[:, i].values),
                            50)

        ax.hist(reconstructed_data[:, i], 
                color=colors[0], 
                alpha=0.5,
                bins=bins,
                label="Generated")

        ax.hist(data.iloc[:, i].values, 
                color=colors[1], 
                alpha=0.5,
                bins=bins,
                label=data.columns[i])

        ax.set_title(f"Reconstruction of {data.columns[i]}")
        ax.legend()

    # Remove empty subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes.flatten()[j])

    # Display the subplots
    plt.show()

def plot_convergences(all_reconstructions, eval_data, descending=False, save=False, model="dense"):
	eval_data_np = eval_data.values
	for i in range(eval_data.shape[1]):
		rec_i = np.asarray([recs[:,i] for recs in all_reconstructions])
		eval_i = eval_data_np[:,i]
		plot_convergence(rec_i, eval_i, f"Convergence {eval_data.columns[i]}", descending, save, model)

def plot_convergence(distri_epochs, target, title="", descending=False, save=False, model="dense"):

	if target.ndim == 1:
		target = target[None,:]

	all_data = np.append(distri_epochs,target,axis=0)
	#determine the number of series
	nseries = len(all_data)
	#define the colormap 
	my_cmap = plt.cm.inferno.reversed()
	#define the yticks, i.e., the column numbers
	yticks = np.arange(nseries)

	if descending:
		all_data = np.flip(all_data,axis=0)
		my_cmap = plt.cm.inferno

	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(projection="3d")

	xbins = np.linspace(all_data.min(), all_data.max(), 100)
	xcenter = np.convolve(xbins, np.ones(2), "valid")/2
	xwidth = np.diff(xbins)

	#calculate now the histogram and plot it for each column
	for i, ytick in enumerate(yticks):

		#extract the current column from your df by its number
		col =  all_data[ytick]

		#determine the histogram values
		#histvals, edges = np.histogram(col, bins="auto")
		histvals, _ = np.histogram(col, bins=xbins)

		#calculate the center and width of each bar
		#obviously not necessary to do this for each column if you always have the same bins 
		#but if you choose for np.histogram other parameters, the bins may not be the same for each histogram
		
		#xcenter = np.convolve(edges, np.ones(2), "valid")/2
		#xwidth = np.diff(edges)

		#plot the histogram as a bar for each bin
		#now with continuous color mapping and edgecolor, so we can better see all bars
		couleur = my_cmap(i/nseries) if i!=(not descending)*(nseries-1) else "black"
		alph = 0.5 if i!=(not descending)*(nseries-1) and i!=0 else 1
		ax.bar(left=xcenter, height=histvals, width=xwidth, zs=ytick, zdir="y", color=couleur, alpha=alph, edgecolor="grey", linewidth=0.3)

	ax.set_xlabel("output range")
	ax.set_ylabel("epochs" if not descending else "epochs reversed")
	ax.set_zlabel("count")

	#label every other column number
	ytick = nseries//10
	ax.set_yticks(yticks[::ytick])
	plt.title(title)
	plt.tight_layout()
	if save:
		title = title.lower().replace(" ","_")
		plt.savefig(f"{model}_{title}.pdf")  
		
	plt.show()
    
# testing function
def eval_VAE(vae_model, data):
	"""Compute the AD and KE score

	Parameters
	----------
	vae_model : the one we will use to get encoded version of data
	data : the data we want to reproduce

	Returns
	-------
	the 2 metrics
	"""
	dim_data = data.ndim
	ad = None
	ke = None
	
	if isinstance(data, pd.DataFrame):
		to_reconstruct = torch.tensor(data.values[None,:,:]).to(torch.float32)
	else:
		to_reconstruct = torch.tensor(data[None,:,:]).to(torch.float32)

	reconstructed_data = vae_model.reconstructe(to_reconstruct)
        
	reconstructed_data = reconstructed_data.cpu().detach().numpy().squeeze()
	to_reconstruct = to_reconstruct.cpu().detach().numpy().squeeze()

	ad = anderson_evaluation(reconstructed_data, to_reconstruct)
	#ad = anderson_ksamp([to_reconstruct.flatten(), reconstructed_data_np.flatten()]).statistic
	if dim_data > 1:
		ke = kendall_evaluation(reconstructed_data, to_reconstruct)

	return ad, ke, reconstructed_data


def anderson_evaluation(generated_input, real_input):
	generated = generated_input.copy() if isinstance(generated_input, pd.DataFrame) else generated_input
	real = real_input.copy() if isinstance(real_input, pd.DataFrame) else real_input

	n = len(real)

	if generated.ndim == 1:
		generated = generated[:, None]
	if real.ndim == 1:
		real = real[:, None]

	num_markets = generated.shape[-1]
	list_ad = []
	for i in range(num_markets):
		list_ad.append(anderson_ksamp([generated[:,i], real[:,i]]).statistic)

	return np.mean(list_ad)

def anderson_evaluation_old(generated_input, real_input):
    generated = generated_input.copy() if isinstance(generated_input, pd.DataFrame) else generated_input
    real = real_input.copy() if isinstance(real_input, pd.DataFrame) else real_input

    n = len(real)
    num_markets = generated.shape[-1]

    if generated.ndim == 1:
        generated = generated[:, None]
    if real.ndim == 1:
        real = real[:, None]
    
    
    u = np.zeros((n, num_markets))
    w = np.zeros(num_markets)
    
    for d in range(num_markets):
        sorted_generated = np.sort(generated[:, d])
        
        for i in range(1, n+1):
            u[i-1][d] = (len(real[:, d][real[:, d]<=sorted_generated[i-1]]) +1) / (n+2)
        
        sum_logs = 0
        for i in range(1, n+1):
            sum_logs += (2*i-1)*(np.log(u[i-1][d]) + np.log(1-u[n-i][d]))
        
        w[d] = -1 * (n + sum_logs/n)
    
    return np.mean(w)

def kendall_evaluation(generated_input, real_input):
    generated = generated_input.copy() if isinstance(generated_input, pd.DataFrame) else generated_input
    real = real_input.copy() if isinstance(real_input, pd.DataFrame) else real_input
    
    n = len(real)
    d = generated.ndim # not used
    
    z_real = np.zeros(n)
    z_generated = np.zeros(n)
    
    for i in range(n):
        arr_real = np.tile(real[i], (n-1, 1)) # array where we repeat the row [X_i^1, ..., X_i^d] n-i times
        arr2_real = np.delete(real, (i), axis=0)
        # we extract the max of each column of the difference
        # useful because the condition a1<b1 and a2<b2 and... ==> a1-b1<0 and a2-b2<0 and... can be resumed as max(ai-bi)<0
        result_real = (arr2_real - arr_real).max(axis=1)
        z_real[i] = len(result_real[result_real<0])

        arr_generated = np.tile(generated[i], (n-1, 1)) # array where we repeat the row [X_i^1, ..., X_i^d] n-i times
        arr2_generated = np.delete(generated, (i), axis=0)
        result_generated = (arr2_generated - arr_generated).max(axis=1)
        z_generated[i] = len(result_generated[result_generated<0])
        
    z_real /= (n-1)
    z_generated /= (n-1)
    
    z_real = np.sort(z_real)
    z_generated = np.sort(z_generated)
    
    return np.mean(np.abs(z_real - z_generated))