from keras import backend as K

def exploss(y_true, y_pred): 
	return K.exp(K.abs(y_true - y_pred)) - 1

def logloss(y_true, y_pred): 
	return -K.log(1 - K.abs(y_true - y_pred))