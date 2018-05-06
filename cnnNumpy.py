import numpy as np
def convolution(X,F,S,P,W,B):
	"""X: Input signal,F: Filter dimensions, S: Stride, P: padding, W: Netwrok Weights, B: Bias"""
	X=np.pad(X,[(P,P),(P,P),(0,0)],'constant',constant_values=0)
	a=X.shape[0]
	b=F[0]
	c=((a-b)/S)+1#output Dimensions
	v=a*1.0
	n=b
	m=((v-n)/S)+1
	if((m).is_integer() !=True or X.shape[2] != F[2]): #Return incompatible for convolution operation
		print ("Invalid Inputs in Conv Layer")
		return 0,0
	else:
		out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
		out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
		im2col=X[out.flatten(),out1.flatten()].reshape(1,c*c*b*b*X.shape[2]).reshape(b*b*X.shape[2],c*c,order="F")
		output_volume=np.dot(W,im2col).reshape(c,c,W.shape[0])+B
		return output_volume,im2col
import numpy as np				
def maxpool(X,F,S):
	a=X.shape[0]
	b=F[0]
	c=((a-b)/S)+1
	v=a*1.0
	n=b
	m=((v-n)/S)+1
	if((m).is_integer() !=True or X.shape[2] != F[2]): 
		print ("Invalid Inputs in Pool Layer")
		return 0
	#l=np.zeros((b*b,c*c*X.shape[2]))
	out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
	out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
	im2col=X[out.flatten(),out1.flatten()].reshape(1,c*c*b*b*X.shape[2]).reshape(b*b*X.shape[2],c*c,order="F")
	z=im2col.reshape(b*b*c*c*X.shape[2],1,order="F").reshape(c*c,b*b,X.shape[2],order="C")
	out=np.argmax(z,axis=1)	
 	out_vol=z[xrange(c*c),out[:,0]]
 	#return out_vol.reshape(c,c,X.shape[2]),out[:,0],c,z.shape

def col2im(im2col,P,F,X,S): # X: list having the dimesions of the output signal or the corresponding signal to im2col
	X[0]=X[0]+2*P
	X[1]=X[1]+2*P
	a=X[0]
	b=F[0]
	c=((a-b)/S)+1#output Dimensions
	im2col=im2col.reshape(1,c*c*b*b*X[2],order="F").reshape(c*c*b*b,X[2],order="C")
	out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
	out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
	Xa=np.empty(((X[0],X[1],X[2])))
	Xa[out.flatten(),out1.flatten()]=im2col[xrange(im2col.shape[0])]
	return Xa[P:X[0]-P,P:X[0]-P]
	
def return_padded(X,P,dim):
	if(dim == 1):
		return np.pad(X,P,'constant',constant_values=0)
	elif(dim == 2):
		return np.pad(X,[(P[0],P[0]),(P[1],P[1])],'constant',constant_values=0)
	elif(dim == 3):	
		return np.pad(X,[(P[0],P[0]),(P[1],P[1]),(P[2],P[2])],'constant',constant_values=0)
def im2col(X,F,S,P):
	"""X: Input signal,F: Filter dimensions, S: Stride, P: padding, W: Netwrok Weights, B: Bias"""
	X=np.pad(X,[(P,P),(P,P),(0,0)],'constant',constant_values=0)
	a=X.shape[0]
	b=F[0]
	c=((a-b)/S)+1#output Dimensions
	v=a*1.0
	n=b
	m=((v-n)/S)+1
	if((m).is_integer() !=True or X.shape[2] != F[2]): #Return incompatible for convolution operation
		print ("Invalid Inputs for im2col")
		return 0,0
	else:
		out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
		out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
		im2col=X[out.flatten(),out1.flatten()].reshape(1,c*c*b*b*X.shape[2]).reshape(b*b*X.shape[2],c*c,order="F")
		return im2col

def maxpool_backward(pool_l1,F,X,argmax,c,z,S):
	b=F[0]
	m=np.zeros(((z[0],z[1],z[2])))#make Zero Image
	m[xrange(c*c),argmax]=pool_l1.reshape(pool_l1.shape[0]*pool_l1.shape[1],X[2])[xrange(c*c)]#populate it With gradients
	im2col=m.reshape(b*b*c*c*X[2],1,order="C").reshape(F[0]*F[1]*F[2],c*c,order='F')#Turn it to Im2col Format
	out=col2im(im2col,0,F,X,S)#Turn Im2col to Image
	return out #Return Image		

def conv_backward():
	pass
